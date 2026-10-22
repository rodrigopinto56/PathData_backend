# 03 — Diccionario de datos

## 1. Fuentes

| Fuente (id interno) | Kaggle | Archivo | Filas | Grano | Uso |
|---|---|---|---|---|---|
| `loan_default_risk` | `himelsarder/loan-default-risk-prediction-dataset` | `loan_default_risk_dataset.csv` | 300 | Un préstamo | Entrenar el modelo de riesgo (tiene el target real) |
| `personal_finance_ml` | `miadul/personal-finance-ml-dataset` | `synthetic_personal_finance_dataset.csv` | 32,424 | Un cliente | Cartera a calificar; base de Gold y KPIs |
| `credit_risk` (S14, opcional) | `laotse/credit-risk-dataset` | `credit_risk_dataset.csv` | ~32.5 K | Un préstamo | Entrenar el modelo predictivo (tiene `loan_status`) |
| `transacciones_cliente_demo` (S14, opcional) | `ramyapintchy/personal-finance-data` | `personal_finance_transactions.csv` | 1,500 | Una transacción | Pronóstico y estabilidad del cliente demo |

Checksums y fecha de descarga: `data/raw/manifest.json`.

> Las fuentes **no comparten una llave común**: el modelo se entrena con una y se aplica a la otra
> mapeando conceptos equivalentes (ver [06](06_decisiones_y_deuda_tecnica.md)).

## 2. Bronze

Copia fiel del CSV en Parquet. La ingesta valida que existan estas columnas mínimas:

| Fuente | Columnas obligatorias |
|---|---|
| `loan_default_risk` | `Retirement_Age`, `Debt_Amount`, `Monthly_Savings`, `Loan_Default_Risk` |
| `personal_finance_ml` | `user_id`, `monthly_income_usd`, `has_loan`, `loan_type`, `loan_amount_usd`, `debt_to_income_ratio`, `credit_score` |

**`_ingestion_metadata.json`** (una por partición):

| Campo | Descripción |
|---|---|
| `fuente` | Id interno de la fuente |
| `kaggle_id` / `sha256_origen` | Linaje: qué dataset y qué versión exacta (del manifest) |
| `filas_ingeridas` | Filas escritas |
| `fecha_logica_particion` | Fecha lógica de Airflow (define la carpeta) |
| `fecha_ejecucion_real` | Cuándo corrió realmente |
| `backend` | `minio` o `local` |

## 3. Silver

Contrato completo: `config/silver_schema.yml`. Registros que no cumplen → `quarantine/` con `motivo_cuarentena`.

### 3.1 `loan_default_risk`

| Columna | Tipo | Nulable | Regla |
|---|---|---|---|
| `Retirement_Age` | double | No | > 0 |
| `Debt_Amount` | double | No | ≥ 0 |
| `Monthly_Savings` | double | No | ≥ 0 |
| `Loan_Default_Risk` | integer | No | ∈ {0, 1} (target del modelo) |

Deduplicación: **fila completa** (no hay columna identificadora).

### 3.2 `personal_finance_ml`

| Columna | Tipo | Nulable | Regla / nota |
|---|---|---|---|
| `user_id` | string | No | Único (llave de deduplicación) |
| `age`, `gender`, `region`, `employment_status` | — | — | Atributos del cliente → `dim_cliente` / `dim_region` |
| `monthly_income_usd` | double | No | ≥ 0 |
| `monthly_expenses_usd` | double | — | Gasto mensual |
| `savings_usd` | double | — | Ahorro acumulado (cola larga: llega a cientos de miles) |
| `has_loan` | string | No | ∈ {Yes/Si, No} |
| `loan_type` | string | Sí | Nulo **solo** si `has_loan = No`; lo contrario → cuarentena |
| `loan_amount_usd` | double | — | Monto del préstamo |
| `debt_to_income_ratio` | double | Sí | Deuda total vs. ingreso (mediana real 1.35 → **no** es el pago mensual) |
| `credit_score` | integer | No | Entre 300 y 850 |
| `monthly_emi_usd`, `loan_interest_rate_pct`, `loan_term_months` | double | — | *Opcionales*: si existen, se usan para el endeudamiento mensual |
| `education_level`, `job_title` | string | — | *Opcionales*: se pasan a `dim_cliente` si existen |

Deduplicación: **por `user_id`**.

### 3.3 Archivos de control

| Archivo | Contenido |
|---|---|
| `_calidad_reporte.json` | `exito_global`, expectativas totales/exitosas/fallidas (con detalle), filas evaluadas |
| `_silver_metadata.json` | `version_dataset: silver_v1`, fecha de congelamiento, resumen de calidad |

### 3.4 `credit_risk` (S14)

| Columna | Regla |
|---|---|
| `loan_status` | ∈ {0, 1}, sin nulos (target) |
| `person_age` | entre 18 y 100 (el dataset trae edades imposibles) |
| `person_income`, `loan_amnt` | > 0 |
| `loan_percent_income` | entre 0 y 1 |
| `person_emp_length` | nulo o entre 0 y 60 |
| `loan_int_rate` | nulo o entre 0 y 40 |
| `loan_grade` | ∈ {A … G} |

Deduplicación por fila completa. El resto de columnas (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`, `cb_person_cred_hist_length`) se conserva.

### 3.5 `transacciones_cliente_demo` (S14)

Columnas renombradas en Silver: `Date → fecha`, `Transaction Description → descripcion`, `Category → categoria`, `Amount → monto`, `Type → tipo`.
Reglas: `fecha` válida, `monto` > 0, `tipo` ∈ {Income, Expense}. Deduplicación por fila completa.

## 4. Gold — modelo estrella

Diseño: `config/gold_schema.yml` → `modelo_dimensional`. Implementación: `scripts/modelo_dimensional.py`.

### 4.1 Dimensiones

| Tabla | PK | Columnas | Nota |
|---|---|---|---|
| `dim_fecha` | `fecha_id` (int `YYYYMMDD`) | `fecha, anio, trimestre, mes, dia, dia_semana (1=lunes), semana_anio` | Una fila por partición |
| `dim_cliente` | `user_id` | `age, rango_edad, gender, employment_status` (+ `education_level, job_title` si existen) | `rango_edad`: 18-24, 25-34, 35-44, 45-54, 55-64, 65+ |
| `dim_region` | `region_id` | `region` | Llave asignada en orden alfabético (determinista) |
| `dim_banda_credito` | `banda_id` | `rango_credit_score, score_min, score_max` | 0 Sin clasificar, 1 Bajo (300–579), 2 Regular (580–669), 3 Bueno (670–739), 4 Muy bueno (740–799), 5 Excelente (800–850) |
| `dim_tipo_prestamo` | `tipo_prestamo_id` | `tipo_prestamo` | Incluye el miembro "Sin prestamo" → la FK nunca es nula |

### 4.2 `fact_posicion_financiera`

Grano: **un renglón por cliente por fecha de partición.**

| Grupo | Columna | Descripción |
|---|---|---|
| Llaves | `fecha_id, user_id, region_id, banda_id, tipo_prestamo_id` | FK a cada dimensión |
| Base | `monthly_income_usd, monthly_expenses_usd, savings_usd` | Desde Silver |
| Base | `tiene_prestamo` | 0/1 normalizado desde `has_loan` |
| Base | `loan_amount_usd` | 0 si no tiene préstamo (monto expuesto, EAD) |
| Base | `debt_to_income_ratio, credit_score` | Desde Silver |
| Opcionales | `loan_term_months, monthly_emi_usd, loan_interest_rate_pct` | Si existen en Silver |
| Modelo | `riesgo_score` | Probabilidad de impago (PD), 0–1. Con el modelo S15: calibrada a la cartera y **nula si el cliente no tiene préstamo** |
| Modelo | `es_riesgo_alto` | 1 si la PD supera el umbral del modelo vigente (S15: 3 × tasa objetivo = 15%; S11: 0.5) |
| Derivada | `flujo_libre_mensual_usd` | ingreso − gasto |
| Derivada | `tasa_ahorro` | (ingreso − gasto) / ingreso |
| Derivada | `ratio_gasto_ingreso` | gasto / ingreso |
| Derivada | `meses_cobertura_ahorro` | ahorro / gasto |
| Derivada | `perdida_esperada_usd` | PD × EAD × LGD (LGD = 45%) |
| Negocio | `pago_mensual_deuda_usd` | Mensualidad real o calculada por amortización |
| Negocio | `ratio_endeudamiento_mensual` | pago mensual / ingreso |
| Negocio | `indice_riesgo_consolidado` | 0–100 (ver [04](04_kpis.md)) |

Divisiones entre 0 (ingreso o gasto = 0) quedan como nulo, no como infinito.

### 4.3 Tablas de consumo y KPIs

| Tabla | Grano | Contenido |
|---|---|---|
| `gold_clientes_riesgo` | Cliente | Perfil + `riesgo_score` + `riesgo_flag` (Alto / Bajo / Sin préstamo). Con el modelo S15 además: `puntaje_riesgo`, `grado_credito_proxy` y `razones_riesgo`. |
| `gold_metricas_por_segmento` | Región × banda de score | Clientes, ingreso y ahorro promedio, % con préstamo, DTI promedio, % riesgo alto. |
| `gold_kpis_financieros` | KPI × fecha (formato largo) | `fecha, categoria, kpi, tipo_indicador (KPI/Metrica), valor, unidad, umbral, meta, brecha, avance_meta_pct, estado, descripcion` |
| `gold_kpis_por_segmento` | Dimensión × segmento × fecha | Clientes, PD, % riesgo alto, score, préstamos, exposición, pérdida esperada, ingreso, tasa de ahorro (+ IC 95%), % deficitarios, endeudamiento mensual, índice consolidado, % semáforo alto. |

Dimensiones de `gold_kpis_por_segmento`: `region, rango_credit_score, tipo_prestamo, rango_edad, employment_status, perfil_ahorro, nivel_riesgo_consolidado`.

### 4.4 Tablas de IA (S14–S15)

| Tabla | Grano | Contenido |
|---|---|---|
| `gold_features_entrenamiento` | Préstamo de `credit_risk` | Features candidatas + `impago` + `particion_modelo` (entrenamiento/validacion/prueba) |
| `reporte_seleccion_variables` | Variable candidata | `iv`, `fuerza`, `decision`, `razon`, `vif` |
| `gold_features_cartera` | Cliente de la cartera | Mismas features (sin alinear) + `tiene_prestamo`, `genero`, `edad` (solo auditoría) |
| `gold_pronostico_cliente_demo` | Mes × serie | `serie` (ingreso/gasto/ahorro), `tipo` (historico/pronostico), `valor`, `limite_inferior`, `limite_superior`, `modelo` |
| `gold_transacciones_mensuales` | Mes × tipo × categoría | `monto`, `transacciones` |
| `_features_metadata/<fecha>.json` | — | Versión del dataset, filas, tasa de impago por partición, features seleccionadas/excluidas, cuantiles de referencia, calidad |
| `_pronostico_metadata/<fecha>.json` | — | Comparación de modelos (MAE), modelos elegidos, estabilidad |
| `gold_simulacion_cartera` (S16) | Escenario | `perdida_esperada`, `var_95/99/99_9` (y `_pct`), `expected_shortfall_99`, `capital_economico`, `prob_superar_reservas_pct`, `prob_superar_colchon_pct`, `exposicion_total`, `prestamos`, `pd_promedio_pct`, `lgd`, `simulaciones` |
| `gold_simulacion_cartera_distribucion` (S16) | Escenario × rango de pérdida | `desde`, `hasta`, `anos`, `pct_anos` (histograma de 40 rangos) |
| `gold_simulacion_cliente_demo` (S16) | Escenario × mes | Percentiles de saldo `p5`, `p25`, `p50`, `p75`, `p95` y `prob_saldo_negativo_pct` |
| `gold_simulacion_cliente_demo_resumen` (S16) | Escenario | `saldo_inicial`, `prob_quedarse_sin_ahorro_pct`, `mes_mediano_agotamiento`, `saldo_final_mediana`, `flujo_mensual_promedio` |
| `_simulacion_metadata/<fecha>.json` (S16) | — | Configuración usada (escenarios, semilla), resumen por escenario y duración |

### 4.5 `_gold_metadata/<YYYY-MM-DD>.json`

| Campo | Contenido |
|---|---|
| `modelo` | `origen` (`campeon_s15` o `modelo_s11`); S15: versión, AUC, Gini, KS, Brier, umbral de riesgo alto; S11: accuracy, AUC, coeficientes |
| `modelo_estrella` | filas por tabla |
| `integridad_modelo_estrella` | `exito_global` + detalle de los 11 chequeos |
| `fuente_pago_mensual` | `monthly_emi_usd`, `amortizacion(...)` o `no_disponible` |
| `gold_clientes_riesgo`, `gold_metricas_por_segmento` | filas |

## 5. Log de eventos (`data/reports/pipeline_log.jsonl`)

Un JSON por línea. Campos comunes: `timestamp`, `nivel` (`INFO` / `ALERTA` / `ERROR`), `task`, `fuente`, `particion`.
Campos específicos según la tarea: `filas_leidas`, `duracion_segundos`, `expectativas_exitosas`,
`accuracy_modelo`, `integridad_modelo_estrella`, `kpis_en_alerta`, `kpis_sin_datos`, `fuente_pago_mensual`, etc.
