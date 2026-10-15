# 05 — Operación y troubleshooting

## 1. Requisitos

- Docker Desktop con Docker Compose.
- Python 3.11+ en el host, solo para descargar datos y correr pruebas.
- Cuenta de Kaggle configurada para `kagglehub`.

## 2. Levantar el entorno desde cero

```powershell
# 1. Datos crudos (genera data/raw/*.csv y actualiza manifest.json)
pip install kagglehub
python scripts/downloadRawData.py

# 2. Construir imágenes e inicializar Airflow
docker compose build
docker compose up airflow-init
docker compose up -d
```

| Servicio | URL | Credenciales |
|---|---|---|
| Airflow | http://localhost:8080 | airflow / airflow |
| MinIO (consola) | http://localhost:9001 | admin / password123 |
| Spark UI | http://localhost:8081 | — |
| Dashboard técnico | http://localhost:8501 | — |
| Dashboard ejecutivo | http://localhost:8502 | — |

> Desde la Semana 14, `downloadRawData.py` descarga también `credit_risk` y las transacciones del cliente demo.
> Son **opcionales**: si alguna falla, se avisa y el pipeline las salta.

### 2.1 Configuración manual (una sola vez)

Estas piezas **no están versionadas** y se crean a mano:

1. **Buckets en MinIO**: `bronze-layer` y `silver-layer` (consola de MinIO → *Create bucket*).
   `gold-layer` se crea solo la primera vez que corre `construir_gold`, y `ml-models` la primera vez que se entrena el modelo.
2. **Conexiones en Airflow** (*Admin → Connections*):

   | Conn Id | Tipo | Configuración |
   |---|---|---|
   | `minio_s3_conn` | Amazon Web Services | Access key `admin`, secret `password123`, Extra: `{"endpoint_url": "http://minio:9000"}` |
   | `spark_default` | Spark | Host `spark://spark`, puerto `7077` |

3. **Módulo de auditoría de Bronze**: `ingestionBronzeLayer.py` importa `bronzeAudit`, que vive en
   `plugins/` (carpeta excluida por `.gitignore`). Ver [06 — Deuda técnica](06_decisiones_y_deuda_tecnica.md).

## 3. Ejecutar el pipeline

Orden: **Bronze → Silver → Gold**. Cada DAG procesa la partición de su fecha lógica.

### 3.1 Backfill de una fecha específica

Los DAGs son `@daily` en zona `America/Mexico_City` (UTC−6): cada corrida inicia a las **00:00 de México = 06:00 UTC**.
Por eso la fecha debe llevar la zona horaria explícita:

```powershell
$F = "2026-09-22T00:00:00-06:00"
docker compose exec airflow-scheduler airflow dags backfill ingesta_bronze_dag        -s $F -e $F --reset-dagruns
docker compose exec airflow-scheduler airflow dags backfill transformacion_silver_dag -s $F -e $F --reset-dagruns
docker compose exec airflow-scheduler airflow dags backfill transformacion_gold_dag   -s $F -e $F --reset-dagruns
```

Para ver el resultado de cada tarea en la terminal (sin pasar por el scheduler):

```powershell
docker compose exec airflow-scheduler airflow dags test transformacion_gold_dag "2026-09-22T00:00:00-06:00"
```

### 3.1b Entrenar el modelo (Semanas 14–15)

Requiere `credit_risk` en Silver para la fecha. Orden: Bronze → Silver → **entrenamiento** → Gold.

```powershell
$F = "2026-10-19T00:00:00-06:00"
docker compose exec airflow-scheduler airflow dags test entrenamiento_modelo_dag $F
docker compose exec airflow-scheduler airflow dags test transformacion_gold_dag $F
```

La primera vez, el modelo se registra y se promueve a campeón. Después, cada entrenamiento compara contra el campeón vigente.

### 3.2 Ejecutar Gold y KPIs a mano (depuración)

```powershell
docker compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python construir_gold.py 2026/09/22 && python kpis_financieros.py 2026/09/22"
```

`construir_gold.py` debe correr primero: escribe las tablas `dim_*` y `fact_*` que lee `kpis_financieros.py`.
Pasos sueltos de IA (mismo contenedor y carpeta): `python features.py 2026/10/19`, `python modelo_credito.py 2026/10/19`
y `python prediccion_financiera.py 2026/10/19`.
Para regenerar solo el reporte ejecutivo de una fecha: `python reporte_ejecutivo.py 2026/09/22` (mismo contenedor y carpeta).

### 3.3 Validaciones manuales

```powershell
# Calidad de Silver (Great Expectations, modo CLI)
docker compose exec airflow-scheduler python /opt/airflow/scripts/validar_calidad_silver.py
# Salud de Bronze en disco local (backend=local)
python scripts/bronzeHealth.py
```

## 4. Pruebas

```powershell
pip install pandas pyarrow scikit-learn pytest
pytest tests/
```

58 pruebas en 4 archivos (`test_gold_kpis.py`, `test_reporte_ejecutivo.py`, `test_modelo_credito.py`, `test_prediccion_financiera.py`).
Usan datos sintéticos con el esquema real (`tests/datos_sinteticos.py`) y no necesitan Docker ni descargas. Cubren:
- **Modelo estrella**: tablas completas, integridad referencial, detección de FK huérfanas, grano, miembro "Sin prestamo", normalización y pérdida esperada.
- **KPIs**: catálogo completo, cuadre de la exposición con la tabla de hechos, semáforo de umbrales y cuadre de segmentos contra el total.
- **Dataset y modelo (S14–S15)**: paridad contrato/código, atributos protegidos fuera del modelo, WoE/IV, binning monótono, alineación por cuantiles, calibración, ajuste de tasa base, pruebas de negocio, registro campeón/retador (un retador peor no reemplaza al campeón).
- **Pronóstico (S15)**: series mensuales, backtesting sin fuga del futuro, baseline como piso, banda de incertidumbre, estabilidad.
- **Dashboard ejecutivo**: reglas del diagnóstico, orden de las acciones, foco de riesgo por región, traducciones y reporte HTML seguro y autocontenido.
- **KPIs de negocio**: fuente del pago mensual (mensualidad real, amortización o N/D), índice consolidado en sus extremos, intervalo de confianza y suma del semáforo = 100%.

## 5. Actualizar después de un `git pull`

| Qué cambió | Qué hacer |
|---|---|
| `dags/`, `scripts/`, `config/`, `src/spark/` | Nada: están montados como volumen. `docker compose restart airflow-scheduler airflow-webserver` si un DAG no refleja el cambio. |
| `dashboard/` o `scripts/reporte_ejecutivo.py` | Reconstruir: `docker compose build streamlit streamlit-ejecutivo` y `docker compose up -d streamlit streamlit-ejecutivo` (el código se **copia** a la imagen). |
| `Dockerfile` (dependencias) | `docker compose build` y `docker compose up -d` |

## 6. Troubleshooting (problemas reales encontrados)

| Síntoma | Causa | Solución |
|---|---|---|
| `backfill` responde **"No run dates were found"** | La fecha sin zona se interpreta como 00:00 UTC, que no coincide con el horario del DAG (06:00 UTC) | Pasar la fecha con zona: `2026-09-22T00:00:00-06:00` |
| Dashboard: **`FileNotFoundError`** en `fs.open(...)` | fsspec reutiliza la conexión a MinIO con un listado viejo del bucket | Corregido en `9727747` (`skip_instance_cache=True`, `use_listings_cache=False`). Reconstruir la imagen de Streamlit. |
| Vista 4: **"Todavía no hay KPIs en Gold"** | La partición se construyó antes de existir la tarea `calcular_kpis` | Re-ejecutar `transformacion_gold_dag` para esa fecha |
| `construir_gold` falla con **"errores de integridad"** | FK huérfana o PK duplicada en el modelo estrella | El mensaje lista los chequeos fallidos; revisar duplicados de `user_id` en Silver |
| `extraer_y_validar` falla sin reintentar | Faltan columnas esperadas en el CSV | Volver a descargar con `downloadRawData.py` y revisar el `manifest.json` |
| `descargar_particion_bronze` falla | No existe la partición Bronze de esa fecha | Correr primero `ingesta_bronze_dag` para esa fecha |
| KPI en **`SIN_DATOS`** / "N/D" | No hay columnas para calcularlo (p. ej. pago mensual) o falta historial | Esperado; revisar `fuente_pago_mensual` en `_gold_metadata` |
| `entrenamiento_modelo_dag` en *skipped* | No hay `credit_risk` en Silver para esa fecha | Correr `downloadRawData.py` y la malla Bronze/Silver de la fecha |
| Evento `ALERTA` en `entrenar_y_registrar` | El modelo nuevo no se promovió (peor que el campeón o falló una prueba de negocio) | Ver `razon_promocion` en la Vista 5 / `model_card.json`; el campeón anterior sigue vigente |
| Vista 3 dice "modelo inicial (Semana 11)" | Todavía no hay campeón registrado | Correr `entrenamiento_modelo_dag` y después el DAG de Gold |
| Dashboard ejecutivo: *"se calcularon con una versión anterior del proceso"* | La partición no tiene los KPIs de negocio | Re-ejecutar `transformacion_gold_dag` para esa fecha |
| Vista 1 muestra valores de `nivel` "OTRO" | Eventos con un nivel distinto de INFO/ALERTA (p. ej. `ERROR`) | El dashboard los agrupa y los avisa; revisar `pipeline_log.jsonl` |
