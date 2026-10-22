# 06 — Decisiones técnicas y deuda técnica

## 1. Decisiones técnicas (formato ADR resumido)

| # | Semana | Decisión | Alternativa descartada | Por qué |
|---|---|---|---|---|
| D1 | 4 | MVP con 2 datasets (`loan_default_risk`, `personal_finance_ml`) | Usar los 3 desde el inicio | Menos alcance y un pipeline completo de punta a punta antes de crecer |
| D2 | 5 | `manifest.json` con SHA-256 versionado en Git; los datos no se versionan | Subir los CSV a Git | Linaje y detección de cambios sin inflar el repositorio |
| D3 | 6 | Parquet particionado por **fecha lógica** | CSV / partición por fecha real | Tipos y compresión; backfills reproducibles e idempotentes |
| D4 | 6 | MinIO como data lake | Disco local | API S3 real: el mismo código migra a AWS cambiando el endpoint |
| D5 | 7 | Log JSON Lines compartido por todos los DAGs | Solo logs de Airflow | Una sola fuente de eventos, fácil de leer desde el dashboard |
| D6 | 7 | `auditar_bronze` con `trigger_rule="all_done"` | Regla por defecto (`all_success`) | La auditoría debe correr justo cuando algo falló |
| D7 | 8 | Imagen propia de Airflow con dependencias instaladas | `_PIP_ADDITIONAL_REQUIREMENTS` | Arranques rápidos y reproducibles (recomendación oficial) |
| D8 | 8 | Cuarentena (`valid/` vs `quarantine/`) | Descartar registros inválidos | No se pierde información; el rechazo es auditable |
| D9 | 9 | Contrato `silver_schema.yml` → expectativas de GE generadas | Declarar reglas en código y en GE | Una sola fuente de verdad para las reglas |
| D10 | 9 | `validar_calidad` reporta pero **no bloquea** | Detener el pipeline si falla | Primera iteración del framework: evitar frenar toda la malla por una regla nueva |
| D11 | 10 | Dashboard con capa de datos separada (`data_loader.py`) | Todo en `app.py` | Se puede probar y cambiar de herramienta sin tocar la interfaz |
| D12 | 11 | Gold con **pandas**, no Spark | Otro job de Spark | ~32 K filas: Spark solo agrega latencia y complejidad |
| D13 | 11 | Regresión logística con 2 features | Modelos más complejos | 300 filas de entrenamiento: menos sobreajuste y coeficientes interpretables |
| D14 | 11 | Z-score **dentro de cada dataset** | Comparar dólares absolutos | Las fuentes no comparten escala (ahorro mensual vs. ahorro acumulado) |
| D15 | 11 | Excluir `Retirement_Age` | Mapearla a `age` | Son conceptos distintos; no se inventan equivalencias |
| D16 | 11 | Esquema estrella + validación de integridad antes de escribir | Solo tablas planas | Normalización pedida por el mentor; no publicar un modelo roto |
| D17 | 11 | Llaves surrogadas en orden alfabético | IDs aleatorios | Deterministas: la misma entrada da las mismas llaves |
| D18 | 12 | KPIs en **formato largo** con umbral y estado | Una columna por KPI | Agregar un KPI no cambia el esquema; el semáforo viaja con el dato |
| D19 | 12 | Alertas de KPI como `INFO` en el log | Registrarlas como `ALERTA` | Una alerta de negocio no es una falla del pipeline |
| D20 | 12 | Medianas para montos con cola larga | Promedios | El promedio se distorsiona por pocos valores extremos |
| D21 | 12 | Endeudamiento con pago mensual (mensualidad o amortización) | Usar `debt_to_income_ratio` | Esa columna mide deuda total (mediana 1.35), no el pago mensual |
| D22 | 12 | `SIN_DATOS` explícito para la varianza | Simular series mensuales | No presentar datos inventados como reales |
| D23 | 14 | Entrenar con `credit_risk` (etiquetado) | Seguir con `loan_default_risk` / inventar etiqueta con reglas | 297 filas no alcanzan; una etiqueta por reglas es circular |
| D24 | 14 | Paridad de features entrenamiento/cartera | Usar todas las variables con IV alto | Una variable que no existe al calificar rompe el modelo en producción |
| D25 | 14 | Alineación por cuantiles de la cartera | Calificar con valores crudos | PSI > 1: sin alinear, toda la cartera caería en el extremo riesgoso |
| D26 | 14 | Excluir género y edad del modelo; usarlos para auditar | Incluirlos si tienen IV | Equidad y regulación de crédito |
| D27 | 15 | Scorecard salvo que el retador gane por ≥ 0.02 de AUC | Elegir siempre el de mayor AUC | Explicabilidad; diferencias pequeñas no compensan perderla |
| D28 | 15 | Calibrar (Platt) y re-escalar a una tasa objetivo | Usar la probabilidad cruda | El entrenamiento tiene ~22% de impago; una cartera típica, ~5% |
| D29 | 15 | DAG de entrenamiento separado + registro campeón/retador | Re-entrenar en cada corrida diaria | Saber qué modelo está en producción y no degradarlo sin querer |
| D30 | 15 | Pronóstico con modelos simples + backtesting vs. baseline | Modelos complejos (ARIMA, redes) | 60 puntos de historia; un modelo que no le gana al baseline no se usa |
| D31 | 16 | Monte Carlo con modelo de un factor de Vasicek (Basilea) | Impagos independientes; cópulas más complejas | Es el estándar regulatorio; sin correlación se subestima la cola. Se valida contra la fórmula cerrada |
| D32 | 16 | Bootstrap de meses reales para el cliente demo | Simular ingreso y gasto con distribuciones normales | No supone una forma de la distribución y conserva la relación ingreso–gasto del mismo mes |
| D33 | 16 | Sensor "mismo día" propio en vez de `ExternalTaskSensor` | `ExternalTaskSensor` con fecha lógica exacta | Los disparos manuales tienen fecha lógica a la hora del clic y el sensor estándar esperaba una corrida inexistente |

## 2. Limitaciones conocidas

| Limitación | Impacto | Mitigación / siguiente paso |
|---|---|---|
| ~~El modelo solo usa monto y ahorro; no reacciona al credit score~~ | **Resuelto en S15:** grado de crédito (proxy del credit score) como variable, con monotonía verificada | — |
| **Entrenamiento y cartera son poblaciones distintas** | No se puede medir el AUC sobre la cartera (no tiene etiqueta); el riesgo es relativo a la propia cartera | Calibración + tasa objetivo + PSI documentado. Definitivo: datos de impago reales de la misma cartera. |
| ~~El modelo se re-entrena en cada corrida, sin versión~~ | **Resuelto en S15:** DAG de entrenamiento + registro campeón/retador | — |
| **Historial mensual solo de un cliente demo** | La estabilidad y el pronóstico no representan a la cartera | Etiquetado como "cliente demo" en todo el tablero. Definitivo: historial transaccional por cliente. |
| **Grado de crédito de la cartera es un proxy** | Supone que el orden del credit score equivale al orden del grado | Documentado en el contrato de features y en la ficha del modelo. |
| **Datos sintéticos** | 0% deficitarios, regiones perfectamente balanceadas (HHI = 0.200) | Interpretar los KPIs como demostración del pipeline, no como hallazgos de negocio. |
| **Pesos del índice consolidado por criterio experto** | No calibrados contra incumplimientos reales | Ajustables en `PESOS_INDICE_RIESGO`; calibrar cuando existan datos. |
| **LGD fija de 45%** | La pérdida esperada es una aproximación regulatoria | Reemplazar por una LGD observada. |
| **Escenarios de estrés por supuesto de negocio** (×1.5, ×2.5) | No son un pronóstico macroeconómico | Configurables en `config/simulador.yml`; ligarlos a variables macro (desempleo, tasas) en una versión futura. |
| **Ahorro inicial del cliente demo supuesto** (3 meses de gasto) | La probabilidad de quedarse sin ahorro depende de ese supuesto | Ajustable en el dashboard; definitivo: saldo real de la cuenta. |

## 3. Deuda técnica

| Prioridad | Tema | Detalle | Acción sugerida |
|---|---|---|---|
| Alta | **`bronzeAudit` fuera de Git** | `ingestionBronzeLayer.py` importa `bronzeAudit.audit_bronze_layer`, que vive en `plugins/` (ignorada por `.gitignore`). Un clon nuevo del repo no puede cargar el DAG. | Mover el módulo a `scripts/` (o quitar `plugins/*` del `.gitignore`) y versionarlo. |
| Alta | **Credenciales en texto plano** | `admin/password123` en `docker-compose.yml` y como valores por defecto en scripts | Usar un archivo `.env` (ya ignorado) y Airflow Connections/Secrets. |
| Media | Conexiones de Airflow y buckets creados a mano | Pasos manuales para levantar el entorno | Crear conexiones con variables `AIRFLOW_CONN_*` y buckets con un servicio `minio-init`. |
| Media | Código duplicado entre DAGs | `registrar_evento` y la configuración de MinIO se repiten en varios archivos | Extraer un módulo común (p. ej. `scripts/comun.py`). |
| Media | `data/validar_conteos_silver.py` | Copia suelta en `data/`, fuera del flujo | Eliminarla o moverla a `scripts/`. |
| Baja | Fechas de prueba fijas en `validar_calidad_silver.py` | `FECHAS_PRUEBA = ["2026/09/01", "2026/09/02"]` en modo CLI | Recibir la fecha como argumento. |
| Baja | Pruebas solo para Gold e IA | Silver y la ingesta no tienen pruebas unitarias (los limpiadores nuevos de S14 se probaron con Spark local, pero no hay prueba automática) | Agregar pruebas de los limpiadores de Spark con datos sintéticos. |
| Media | Artefactos `joblib` en MinIO | Deserializar un modelo ejecuta código: el bucket `ml-models` debe ser escribible solo por el pipeline | Políticas de MinIO por rol (ver deuda de credenciales). |
| Baja | Versión de scikit-learn sin fijar en `Dockerfile` | Un modelo serializado con una versión puede no cargar con otra | Fijar la versión al estabilizar el modelo. |
| Baja | Aviso de pandas en la Vista 3 | `groupby().apply()` sobre columnas de agrupación (`FutureWarning`) | Usar `include_groups=False` o agregar con `agg`. |
