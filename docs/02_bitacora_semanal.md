# 02 — Bitácora semanal (Semana 4 a Semana 15)

Registro de lo que se construyó cada semana según el plan del proyecto: objetivo, entregable,
archivos, decisiones técnicas y evidencia.

> **Nota sobre los commits:** algunos mensajes de commit usan un número de semana distinto al del
> plan (por ejemplo, el commit "Semana 6" contiene la ingesta de la Semana 5). Esta bitácora sigue el
> **plan oficial**; la columna *Commit* indica dónde está el código en el historial de Git.

| Semana | Fechas | Fase | Tema | Entregable | Commit(s) |
|---|---|---|---|---|---|
| 4 | 3–7 ago | 2. Ingesta y Bronze | Exploración de datasets | Data profiling, diccionario de datos | `1758dd7` (`pathDataBBVA.py`) |
| 5 | 10–14 ago | 2. Ingesta y Bronze | Pipelines de ingesta | Pipeline de carga inicial | `1758dd7` |
| 6 | 17–21 ago | 2. Ingesta y Bronze | Implementación Bronze | Capa Bronze funcional | `1758dd7`, `6ad5242` |
| 7 | 24–28 ago | 2. Ingesta y Bronze | Automatización y monitoreo | DAG de Airflow, logs básicos | `933bc86` |
| 8 | 31 ago–4 sep | 3. Silver y calidad | Transformaciones iniciales | Jobs PySpark | `971b082` |
| 9 | 7–11 sep | 3. Silver y calidad | Validación, deduplicación, framework de calidad | Dataset Silver v1, Great Expectations | `971b082`, `55062e3` |
| 10 | 14–18 sep | 3. Silver y calidad | Dashboard de calidad | Dashboard operativo inicial | `55062e3` |
| 11 | 21–25 sep | 4. Gold y analítica | Diseño analítico y construcción Gold | Modelo analítico, tablas Gold | `162905d`, `88d9791` |
| 12 | 28 sep–2 oct | 4. Gold y analítica | KPIs financieros | KPIs calculados | `88d9791`, `9727747`, `ce3f0ef` |
| 13 | 5–9 oct | 4. Gold y analítica | Dashboard ejecutivo | Dashboard ejecutivo v1 | `6d19947`, `03c6810` |
| 14 | 12–16 oct | 5. IA | Preparación de datasets para IA | Dataset para modelos | ver historial (S14–S15) |
| 15 | 19–23 oct | 5. IA | Modelo predictivo | Modelo ML funcional | ver historial (S14–S15) |

---

## Semana 4 — Exploración de datasets

**Objetivo:** analizar la calidad inicial de los datos e identificar reglas de negocio.

**Qué se hizo**
- `pathDataBBVA.py` descarga tres datasets de Kaggle con `kagglehub`:
  `loan-default-risk-prediction-dataset`, `personal-finance-ml-dataset` y `personal-finance-data`.
- Genera un **reporte de profiling** por dataset con `ydata-profiling` (modo `minimal`) en `reports/*_profile.html`.
- Genera el **esqueleto del diccionario de datos** (`reports/diccionario_datos_esqueleto.csv`): tipo,
  % de nulos, valores únicos y outliers por IQR (1.5 × rango intercuartil) de cada columna, más
  campos vacíos para completar a mano: descripción, regla de negocio candidata y clasificación.

**Hallazgos que definieron reglas posteriores**
- `loan_default_risk` (300 filas) **no tiene columna identificadora** → la deduplicación debe ser por fila completa.
- `personal_finance_ml` (32,424 filas): `user_id` es único → llave de deduplicación.
- `loan_type` es nulo en ~60% de los casos, pero **solo cuando `has_loan = No`** → no es un dato faltante, es una regla de consistencia.
- `debt_to_income_ratio` tiene distribución bimodal con outliers esperados.
- Se decidió trabajar solo con **2 datasets en el MVP**; el tercero quedó en reserva.

**Tecnologías:** Python, pandas, kagglehub, ydata-profiling.

---

## Semana 5 — Construcción de pipelines de ingesta

**Objetivo:** automatizar la carga de datos.

**Qué se hizo**
- `scripts/downloadRawData.py`: descarga los 2 datasets del MVP a `data/raw/` y escribe
  `data/raw/manifest.json` con `kaggle_id`, **checksum SHA-256**, fecha de descarga y número de filas.
- Primera versión de `dags/ingestionBronzeLayer.py` (DAG `ingesta_bronze_dag`) y de `docker-compose.yml` (Airflow + Postgres).
- `.gitignore` que excluye los datos (`data/raw/*.csv`, `data/bronze/`, reportes) pero **sí versiona `manifest.json`**.

**Decisiones técnicas**
- **Manifest con SHA-256:** permite detectar si Kaggle actualizó un dataset y deja trazabilidad
  (linaje) de qué versión exacta se ingirió.
- Los datos no se versionan en Git: se regeneran con el script.

---

## Semana 6 — Implementación de la capa Bronze

**Objetivo:** persistir los datos crudos de forma confiable.

**Qué se hizo**
- El DAG convierte el CSV a **Parquet** y lo guarda particionado por fecha lógica: `<fuente>/YYYY/MM/DD/data.parquet`.
- Cada partición lleva `_ingestion_metadata.json` (fuente, kaggle_id, sha256, filas, fecha lógica, fecha real de ejecución, backend).
- Validación de **esquema de entrada**: si faltan columnas esperadas, la tarea falla con `AirflowFailException` (sin reintentos: reintentar no arregla un CSV mal formado).
- Se integró **MinIO** como data lake compatible con S3 (bucket `bronze-layer`), con backend configurable por variable de entorno (`BRONZE_BACKEND=minio|local`), usando `S3Hook` y la conexión `minio_s3_conn`.
- `scripts/bronzeHealth.py`: auditoría local que detecta particiones incompletas, JSON corruptos y **huecos de fechas** en el rango.

**Decisiones técnicas**
- **Parquet** en lugar de CSV: columnar, comprimido y conserva tipos.
- **Partición por fecha lógica** (no por fecha real): un backfill del día X siempre escribe en la carpeta X → re-ejecuciones reproducibles.
- **MinIO** reproduce localmente la API de S3: el mismo código funcionaría en AWS cambiando solo el endpoint.

---

## Semana 7 — Automatización y monitoreo inicial

**Objetivo:** programar la ejecución automática y tener visibilidad de lo que pasa.

**Qué se hizo** (versión v6 de `ingestionBronzeLayer.py`)
- DAG `@daily` con TaskFlow API; `retries=3` y `retry_delay=5 min`.
- **Logging estructurado** en JSON Lines (`data/reports/pipeline_log.jsonl`): cada tarea registra nivel, fuente, filas, duración y resultado.
- **Alertamiento:** `on_failure_callback=alertar_fallo` registra un evento `ALERTA` cuando una tarea agota sus reintentos.
- **Auditoría automática:** tarea `auditar_bronze` con `trigger_rule="all_done"`, que corre aunque falle una ingesta y valida que todas las particiones tengan sus 2 archivos.

**Evidencia:** backfills de agosto–septiembre con `auditar_bronze` reportando 60+ particiones y 0 problemas (ver `pipeline_log.jsonl`).

---

## Semana 8 — Transformaciones iniciales (Jobs PySpark)

**Objetivo:** limpieza y normalización de datos.

**Qué se hizo**
- Clúster Spark en Docker (`spark` master + `spark-worker`) y `src/spark/hello_spark.py` para
  validar la infraestructura **antes** de escribir lógica de negocio.
- `Dockerfile` propio de Airflow con Java 17 + PySpark + provider de Spark (reemplaza `_PIP_ADDITIONAL_REQUIREMENTS`).
- `src/spark/bronze_to_silver.py`: un limpiador por fuente (`CLEANERS`) que tipa columnas y aplica reglas:
  - `loan_default_risk`: `Loan_Default_Risk ∈ {0,1}`, `Retirement_Age > 0`, `Debt_Amount ≥ 0`, `Monthly_Savings ≥ 0`, sin nulos.
  - `personal_finance_ml`: `user_id` no nulo, `monthly_income_usd ≥ 0` y consistencia `has_loan`/`loan_type`.
- **Cuarentena:** los registros inválidos no se borran; van a `quarantine/` con la columna `motivo_cuarentena`.
- DAG `transformacion_silver_dag` con `SparkSubmitOperator` (driver en Airflow, executors en el clúster).

**Decisión técnica:** cuarentena en vez de descartar → no se pierde información y se puede auditar por qué se rechazó cada registro.

---

## Semana 9 — Validación, deduplicación y framework de calidad

**Objetivo:** resolver inconsistencias y definir reglas de validación declarativas.

**Qué se hizo**
- **Deduplicación** en `bronze_to_silver.py`: por `user_id` en `personal_finance_ml` y por fila completa en `loan_default_risk`, con conteo de duplicados removidos en el log.
- **Idempotencia** en `subir_silver_a_minio`: borra el prefijo antes de subir, así re-ejecutar no deja archivos viejos mezclados.
- `config/silver_schema.yml`: **contrato de esquema**, fuente única de verdad de tipos, nulabilidad, reglas y llave de deduplicación.
- `scripts/validar_calidad_silver.py`: genera **automáticamente** el Expectation Suite de Great Expectations a partir del contrato (no se declaran las reglas dos veces). Se usa desde el DAG (`validar_particion`) o por CLI.
- Tareas nuevas en el DAG: `validar_calidad` (guarda `_calidad_reporte.json`) y `congelar_silver_v1` (guarda `_silver_metadata.json`: **Dataset Silver v1**).

**Evidencia:** 8/8 expectativas cumplidas en ambas fuentes (`data/reporte_calidad_silver.json`); `loan_default_risk` pasa de 300 a 297 filas válidas.

---

## Semana 10 — Dashboard de calidad

**Objetivo:** visualizar métricas de calidad y contar la historia de los datos (*storytelling*).

**Qué se hizo**
- `dashboard/` con **Streamlit + Plotly**, en su propio contenedor (`Dockerfile.streamlit`, puerto 8501).
- `data_loader.py` (capa de datos) separado de `app.py` (interfaz).
- Vista 1 — **Salud de la malla** (desde `pipeline_log.jsonl`) y Vista 2 — **Calidad de datos** (desde `_calidad_reporte.json`, umbral 95%).
- **Resumen ejecutivo** que se arma con los datos actuales: si algo se degrada, el texto cambia solo.
- Paleta de colores fija: estados (verde/rojo), categorías (fuentes) y rampas secuenciales (bandas ordenadas).

---

## Semana 11 — Diseño analítico y construcción Gold

**Objetivo:** definir las métricas financieras y crear las tablas Gold.

**Parte 1 — Modelo de riesgo (`162905d`)**
- `scripts/modelo_riesgo.py`: `LogisticRegression` entrenada con `loan_default_risk`
  (features `Debt_Amount` y `Monthly_Savings`, target `Loan_Default_Risk`) y aplicada a `personal_finance_ml`
  mapeando `loan_amount_usd` y `savings_usd`. Estandarización **z-score dentro de cada dataset**, porque las escalas en dólares no son comparables entre fuentes.
- `Retirement_Age` se excluyó: no tiene equivalente en `personal_finance_ml`.
- Primeras tablas Gold: `gold_clientes_riesgo` y `gold_metricas_por_segmento`. Vista 3 del dashboard.

**Parte 2 — Modelo analítico normalizado (`88d9791`)** *(observación del mentor: faltaban las tablas Gold normalizadas)*
- `scripts/modelo_dimensional.py`: **esquema estrella** con 5 dimensiones (`dim_fecha`, `dim_cliente`,
  `dim_region`, `dim_banda_credito`, `dim_tipo_prestamo`) y la tabla de hechos `fact_posicion_financiera`.
- Medidas derivadas por cliente, calculadas una sola vez: flujo libre, tasa de ahorro, gasto/ingreso, meses de cobertura y pérdida esperada.
- **Validación de integridad** (11 chequeos) antes de escribir; si falla, la tarea falla.
- `config/gold_schema.yml`: diseño del modelo, bandas de credit score y definición de métricas financieras.

**Evidencia:** `_gold_metadata/<fecha>.json` con `integridad_modelo_estrella.exito_global = true`; modelo con AUC 0.86 en datos reales.

---

## Semana 12 — KPIs financieros

**Objetivo:** generar KPIs de riesgo financiero, exposición y comportamiento transaccional.

**Qué se hizo**
- `scripts/kpis_financieros.py` + tarea `calcular_kpis` en `transformacion_gold_dag`.
- **35 KPIs** en 4 familias (37 desde la S15, con la estabilidad), cada uno con unidad, descripción, umbral y estado (`OK`/`ALERTA`/`INFO`/`SIN_DATOS`):
  KPIs de negocio (10), riesgo financiero (7), exposición (10) y comportamiento transaccional (8).
- **KPIs por segmento** (región, banda de score, tipo de préstamo, edad, empleo, perfil de ahorro, nivel de riesgo).
- **KPIs de negocio de la lámina** (`ce3f0ef`):
  - Capacidad de ahorro con **intervalo de confianza 95% por bootstrap**.
  - **Endeudamiento mensual** = pago mensual / ingreso (meta < 36%), con el pago tomado de `monthly_emi_usd` o calculado con la fórmula de amortización.
  - **Índice de riesgo consolidado** 0–100 con semáforo Bajo/Medio/Alto.
  - **Varianza de ingresos/gastos**: definida, pero en `SIN_DATOS` (el dataset no trae historial mensual).
- Vista 4 del dashboard: KPIs de negocio, semáforo, gráficas y tabla por segmento.
- `tests/test_gold_kpis.py`: 17 pruebas unitarias con datos sintéticos.
- Corrección `9727747`: `FileNotFoundError` en el dashboard por la caché de listados de s3fs.

**Evidencia (partición 2026/09/22, datos reales):** AUC 0.86, exposición total $3,246 M, tasa de ahorro mediana 39.9%.

---

## Semana 13 — Dashboard ejecutivo

**Objetivo:** publicar los indicadores para alguien que no es programador (p. ej. un ejecutivo de negocio).

**Qué se hizo**
- `scripts/reporte_ejecutivo.py`: capa de **lenguaje de negocio** sobre los KPIs de Gold:
  - Diagnóstico de la cartera (Saludable / Requiere atención / Crítico) en una frase.
  - Los 4 KPIs de la lámina explicados con palabras ("de cada $100 que gana...").
  - **Acciones sugeridas** por reglas: cada KPI en alerta se traduce en una acción concreta con prioridad
    (p. ej. endeudamiento alto → ofrecer consolidación de deuda).
  - Nombres de segmentos traducidos (Employed → Empleado, North → Norte) y montos legibles ("$3.2 mil millones").
  - Reporte HTML de una página, autocontenido e imprimible.
- `dashboard/ejecutivo.py`: app Streamlit separada (puerto 8502, tema claro, color institucional), en el orden
  en que lee un ejecutivo: diagnóstico → indicadores clave → semáforo → dinero en juego → dónde está el riesgo →
  acciones → clientes prioritarios (descarga CSV) → descarga del reporte.
- Tarea `publicar_reporte_ejecutivo` en `transformacion_gold_dag` ("publicar indicadores"): cada día deja el
  reporte en `gold-layer/reportes_ejecutivos/` y `data/reports/ejecutivo/`.
- `tests/test_reporte_ejecutivo.py`: 10 pruebas (reglas del diagnóstico, orden de acciones, foco de riesgo,
  traducciones, HTML sin recursos externos y con texto escapado). Total del proyecto: 27 pruebas.
- `docs/07_guia_dashboard_ejecutivo.md`: guía de uso sin tecnicismos.

**Decisiones técnicas**
- **Dos dashboards, dos audiencias:** el técnico conserva el detalle (salud de la malla, calidad, 35 KPIs); el
  ejecutivo muestra solo lo que sirve para decidir.
- **Una sola lógica de negocio** compartida por el dashboard y el reporte: nunca se contradicen.
- **Estado = ícono + texto + color** (✓ / ! / ✕), nunca solo color (accesibilidad y daltonismo).
- **Rojo reservado para riesgo:** los botones usan el azul institucional, no el rojo por defecto de Streamlit.
- **Acciones por reglas** (no IA generativa): explicables, auditables y deterministas.
- **Tipografía:** títulos en *Source Serif 4* (serif editorial, como la lámina del proyecto) y texto y cifras
  en *IBM Plex Sans* con números tabulares. El reporte HTML no carga fuentes externas (para verse igual sin
  internet): usa Georgia / Segoe UI si esas fuentes no están instaladas.

---

## Semana 14 — Preparación de datasets para IA

**Objetivo:** un dataset de variables versionado, documentado y validado, igual para entrenar y para calificar.

**Qué se hizo**
- **Dos fuentes nuevas** por toda la malla, con los mismos patrones de siempre (contrato, cuarentena, deduplicación, Great Expectations):
  - `credit_risk`: ~32 K préstamos **etiquetados** (pagó / no pagó). Es la base del modelo. Cuarentena para edades imposibles y antigüedades laborales absurdas.
  - `transacciones_cliente_demo`: 60 meses de transacciones de un cliente. Es la serie de tiempo del pronóstico. Las columnas se normalizan a snake_case en español.
  - Las dos son **opcionales**: si no se descargaron, sus tareas se saltan sin romper el resto.
- **Contrato de features** `config/features.yml`: qué variables, cómo se calculan en cada lado, unidad, rango, monotonía y por qué se excluye cada candidata.
- `scripts/features.py`:
  - *Feature engineering* con **paridad** entrenamiento/cartera (ratios, logaritmos, grado de crédito por cuantiles del credit score).
  - **WoE/IV** con bins monótonos.
  - **Selección** por IV, redundancia (Spearman, VIF), paridad y atributos protegidos.
  - **Partición** 70/15/15 estratificada.
  - **Validación** contra el contrato.
  - **Alineación por cuantiles** de la cartera, con su medición de PSI.
- El generador de expectativas de Great Expectations ahora entiende reglas genéricas (`entre A y B`, `en {...}`), sin cambiar las que ya existían.

**Decisiones técnicas**
- **Paridad antes que cantidad:** se excluyen variables útiles (antigüedad laboral, impago previo) porque no existen en la cartera. Un modelo que usa datos que no tendrá al calificar falla en producción.
- **Género y edad fuera del modelo**, pero dentro de la auditoría de equidad.
- **Alineación por cuantiles:** la cartera y el entrenamiento son poblaciones distintas (PSI > 1 en el ratio préstamo/ingreso). Se compara a cada cliente contra su propia cartera, y la limitación queda documentada.

---

## Semana 15 — Modelo predictivo

**Objetivo:** un modelo de riesgo crediticio entrenado, evaluado, calibrado, versionado y explicable, más una predicción financiera.

**Qué se hizo**
- `scripts/modelo_credito.py`:
  - **Dos candidatos:** scorecard (WoE + logística) y gradient boosting con restricciones de monotonía.
  - **Evaluación:** validación cruzada de 5 pliegues con AUC, **Gini**, KS y Brier.
  - **Calibración:** Platt más ajuste a la tasa de impago objetivo de la cartera (5%).
  - **Controles:** pruebas de sentido de negocio (monotonía) y auditoría de equidad (regla 4/5).
  - **Registro** versionado con regla **campeón/retador**.
  - **Calificación** de la cartera con PD, puntaje, nivel de riesgo y **3 razones por cliente**.
- `dags/entrenamiento_modelo_dag.py`: el entrenamiento se **separa** de la calificación diaria (semanal o manual). `construir_gold` solo carga el campeón y, si no existe, usa el modelo de la Semana 11.
- `scripts/prediccion_financiera.py`: pronóstico de 6 meses de ingreso, gasto y ahorro del cliente demo, con *backtesting* contra baselines y banda de incertidumbre. Alimenta el KPI de **estabilidad**, que deja de estar "Pendiente".
- **KPIs con meta, brecha y % de avance** (pedido del mentor): cada indicador se clasifica como **KPI** (tiene meta) o **Métrica** (contexto).
- **Dashboards:**
  - **Técnico:** nueva Vista 5 con candidatos, calibración, peso de variables, IV, pruebas, equidad, PSI e historial de versiones.
  - **Ejecutivo:** barras de avance, probabilidad de impago y factores por cliente, y la proyección del cliente demo.
- **Pruebas:** 58 en total. Se agregaron pruebas del modelo, de features y del pronóstico, con un generador sintético que tiene el mismo esquema que el dataset real.
- **Documentación:** [08 — Dataset para IA y modelo](08_dataset_ia_y_modelo.md).

**Decisiones técnicas**
- **Scorecard por defecto:** el retador necesita +0.02 de AUC sin peor calibración. En banca lo explicable gana los empates.
- **Entrenamiento ≠ inferencia:** versión registrada, promoción con reglas y evento `ALERTA` si un retador no se promueve.
- **El pronóstico solo usa un modelo si le gana al baseline.** Con los datos reales le gana: −26% de error en ingreso y −37% en gasto.
- **Sin dependencias nuevas:** scikit-learn ya estaba; las series de tiempo se implementaron con numpy.

**Evidencia:** prueba de punta a punta con S3 simulado y con el CSV real de transacciones: Gold con respaldo → dataset → entrenamiento → re-entrenamiento con campeón/retador → Gold con campeón → pronóstico → 37 KPIs → reporte. El dataset de riesgo se probó con datos sintéticos del mismo esquema; sus métricas reales salen al correr el DAG con el dataset descargado.

### Ajustes de robustez (post Semana 15)

Aparecieron al correr la malla con datos reales: el entrenamiento semanal salía *skipped* y los DAGs diarios podían arrancar antes de tener sus datos.

- **Entrenamiento con la partición más reciente** (`features.particion_mas_reciente`): la corrida semanal ya no exige que exista la partición de su fecha lógica exacta. Usa la más reciente de `credit_risk` (y de la cartera) hasta el **fin** de su periodo (`data_interval_end`) y lo registra en `particion_origen`. Nunca usa datos posteriores a ese día.
- **Sensores entre DAGs** (`dags/espera_dags.py`): Silver espera a que Bronze termine con éxito el mismo día, y Gold espera a Silver. Aceptan corridas programadas, backfills y disparos manuales a cualquier hora. Fallan de inmediato si el DAG anterior falló, y se pueden saltar con `{"sin_esperar": true}`.
- **Validación:** los 4 DAGs cargan sin errores en Airflow 2.9.3. El sensor se probó contra la base de Airflow en 9 casos: sin corrida, corriendo, fallida, exitosa a las 23:59, otro día, y frontera de zona horaria. La prueba `test_particion_mas_reciente` cubre la búsqueda de la partición.

## Semana 16 — Simulador financiero (Monte Carlo)

**Objetivo del plan:** implementar Monte Carlo y un simulador de escenarios.

**Qué se hizo**
- `scripts/simulador_montecarlo.py` + `config/simulador.yml`: dos simuladores.
  - **Cartera de crédito:** modelo de un factor de **Vasicek**, el de Basilea. Simula 10,000 años por escenario
    (base, adverso, severo) con la probabilidad de impago de cada cliente del modelo S15. Calcula pérdida esperada,
    **VaR** 95/99/99.9%, *expected shortfall*, capital económico y la probabilidad de rebasar el colchón actual.
  - **Cliente demo:** *bootstrap* de sus meses reales. Simula 5,000 futuros de 12 meses por escenario
    (inflación, recesión, desempleo) y mide la probabilidad de quedarse sin ahorro y el abanico del saldo.
- `transformacion_gold_dag`: nueva tarea `simular_escenarios` después del pronóstico. Escribe 4 tablas Gold y `_simulacion_metadata`.
- **Dashboard ejecutivo:** sección *"Simulador de escenarios"* con dos pestañas.
  - Tabla de escenarios en lenguaje de negocio, histograma de pérdidas con la línea del colchón y abanico del ahorro.
  - **Arma tu propio escenario:** sliders que simulan en vivo. La cartera corre 2,000 años en ~1 s; se compara contra una base simulada igual, para que la diferencia no sea ruido.
- **Pruebas:** 11 nuevas (70 en total), validadas contra la fórmula cerrada de Vasicek/Basilea.
- **Documentación:** [09 — Simulador Monte Carlo](09_simulador_montecarlo.md).

**Decisiones técnicas**
- **Vasicek con correlación de Basilea:** sin correlación los impagos serían independientes y la simulación casi nunca mostraría años de crisis. La correlación hace que en un año malo caigan muchos a la vez.
- **El colchón se mide contra el escenario base:** "reservas + capital" = peor caso 1 de 1,000 de hoy. Así cada escenario responde "¿con qué frecuencia no alcanzaría lo que tenemos?".
- **Mismo código en el DAG y en el dashboard:** el resultado oficial lo calcula el DAG. El dashboard reutiliza las mismas funciones y el mismo YAML para los escenarios propios.
- **Semilla fija:** la misma fecha y configuración dan el mismo resultado (auditable).

**Evidencia**
- La pérdida esperada simulada (5.12% de la exposición) cuadra con el KPI calculado por fórmula (5.10%).
- Con 17,840 préstamos (tamaño real) la simulación tarda unos 13 s.
- Con las transacciones reales, el cliente demo gasta más de lo que gana (unos −$6,800 al mes). Aun en el escenario base agota su ahorro supuesto en 79% de los futuros, lo que es coherente con el pronóstico S15.

## Semana 17 — Asistente IA (consultas en lenguaje natural)

**Objetivo del plan:** prototipo de asistente financiero con Ollama, Llama 3 y LangChain.

**Qué se hizo**
- `scripts/asistente_financiero.py`: responde preguntas como "¿qué región tiene más riesgo?", "¿por qué el cliente U00041 es riesgoso?" o "¿qué pasaría si hay una crisis?" con los datos de Gold.
  - **Enrutador por reglas:** decide qué datos hacen falta.
  - **Consultas a Gold:** reutilizan la lógica del tablero.
  - **Llama 3.2 vía Ollama y LangChain:** solo redacta la respuesta.
  - **Verificador:** marca cualquier cifra que no esté en los datos.
- **Modo básico:** si la IA no está disponible, el asistente muestra los datos sin redactar y el tablero no se rompe.
- `docker-compose.yml`: servicios `ollama` (servidor local) y `ollama-init` (descarga el modelo una vez).
- **Dashboard ejecutivo:** sección *"Pregúntale al asistente"* con preguntas de ejemplo, chat, los datos consultados y el aviso de cifras no verificadas.
- **Pruebas:** 24 nuevas (94 en total), con un modelo falso de LangChain y un servidor que imita la API de Ollama.
- **Documentación:** [10 — Asistente financiero con IA](10_asistente_ia.md).

**Decisiones técnicas**
- **IA local y gratuita:** Ollama no cobra por uso y los datos de clientes no salen de nuestros servidores.
- **La IA redacta, el código busca los datos:** un modelo pequeño elige mal herramientas y puede inventar números. Así, lo que debe ser exacto lo hace código probado.
- **Llama 3.2 3B por defecto:** corre en CPU con ~4 GB de RAM. Con más memoria se cambia a `llama3.1:8b` sin tocar código.

**Ajuste tras la primera prueba con Llama real:** el modelo de 3B copió el formato, calificó como "crítico" un indicador que sí cumplía su meta y se negó a responder teniendo los datos. Se agregaron cuatro mejoras:
- control de calidad con reintento y respaldo de datos exactos;
- un ejemplo de buena respuesta en el prompt;
- nombres de negocio sin fórmulas en los KPIs;
- la evaluación automática (`scripts/evaluar_asistente.py`, 30 preguntas, meta ≥ 84%).

**Evidencia:** prueba de punta a punta con datos sintéticos (pipeline → Gold → tablero → chat) en modo básico y contra un servidor que imita la API de Ollama. La redacción con el modelo real se valida en la máquina del equipo, porque en el entorno de desarrollo la descarga de modelos estaba bloqueada.
