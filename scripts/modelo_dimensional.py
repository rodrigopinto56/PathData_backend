"""
scripts/modelo_dimensional.py

Semana 11 (cierre): modelo analitico de la capa Gold en esquema estrella.

Hasta ahora Gold tenia dos tablas "planas" (gold_clientes_riesgo y
gold_metricas_por_segmento): utiles para el dashboard, pero cada una
repite los mismos atributos descriptivos (region, banda de score, tipo de
prestamo...) en cada renglon. Aqui se normalizan en un esquema estrella:

    dim_fecha ─┐
    dim_cliente ─┤
    dim_region ──┼── fact_posicion_financiera
    dim_banda_credito ─┤
    dim_tipo_prestamo ─┘

  - Las dimensiones guardan el "quien / donde / que" UNA sola vez, con su
    propia llave (PK).
  - La tabla de hechos guarda solo llaves foraneas (FK) + medidas
    numericas. Es la tabla sobre la que se calculan los KPIs de la
    Semana 12 (ver scripts/kpis_financieros.py).

Ver config/gold_schema.yml, seccion modelo_dimensional, para el diseno
completo (grano, llaves y columnas de cada tabla).

Este modulo es puro pandas, sin lectura/escritura a MinIO: recibe
DataFrames y regresa DataFrames. La E/S vive en construir_gold.py. Asi
se puede probar sin levantar el stack (ver tests/test_gold_kpis.py).
"""

from __future__ import annotations

import pandas as pd

# Ver config/gold_schema.yml, seccion bandas_credit_score
BANDAS_CREDIT_SCORE = [
    (1, 300, 579, "Bajo"),
    (2, 580, 669, "Regular"),
    (3, 670, 739, "Bueno"),
    (4, 740, 799, "Muy bueno"),
    (5, 800, 850, "Excelente"),
]
BANDA_SIN_CLASIFICAR = (0, None, None, "Sin clasificar")

# Ver config/gold_schema.yml, seccion modelo_dimensional.dim_cliente
RANGOS_EDAD = [
    (0, 24, "18-24"),
    (25, 34, "25-34"),
    (35, 44, "35-44"),
    (45, 54, "45-54"),
    (55, 64, "55-64"),
    (65, 200, "65+"),
]

# Ver config/gold_schema.yml, seccion parametros_financieros.lgd
# Loss Given Default: % del monto que se pierde si el cliente incumple.
# 45% es el valor supervisor de Basilea II (enfoque IRB basico) para
# deuda senior sin garantia -- no tenemos datos de recuperacion propios.
LGD = 0.45

SIN_PRESTAMO = "Sin prestamo"
VALORES_SI = {"yes", "si", "sí", "true", "1"}

# Columnas de personal_finance_ml que NO usa el modelo base pero que, si
# vienen en Silver, se pasan a la tabla de hechos / dim_cliente. Se
# tratan como opcionales para que el pipeline no truene si una version
# del dataset no las trae.
MEDIDAS_OPCIONALES = ["loan_term_months", "monthly_emi_usd", "loan_interest_rate_pct"]
ATRIBUTOS_CLIENTE_OPCIONALES = ["education_level", "job_title"]


def _fecha_id(fecha: str) -> int:
    """'2026/09/22' -> 20260922 (llave entera de dim_fecha)."""
    return int(fecha.replace("/", "").replace("-", ""))


def _rango_edad(edad) -> str:
    if pd.isna(edad):
        return "Sin dato"
    for minimo, maximo, etiqueta in RANGOS_EDAD:
        if minimo <= edad <= maximo:
            return etiqueta
    return "Sin dato"


def banda_credit_score_id(score) -> int:
    if pd.isna(score):
        return BANDA_SIN_CLASIFICAR[0]
    for banda_id, minimo, maximo, _ in BANDAS_CREDIT_SCORE:
        if minimo <= score <= maximo:
            return banda_id
    return BANDA_SIN_CLASIFICAR[0]


def tiene_prestamo(serie: pd.Series) -> pd.Series:
    """has_loan viene como 'Yes'/'No' en el CSV original, pero el contrato
    de Silver (silver_schema.yml) acepta tambien 'Si'. Se normaliza a 0/1
    para que la tabla de hechos tenga una medida sumable."""
    return serie.astype(str).str.strip().str.lower().isin(VALORES_SI).astype(int)


def _dimension_catalogo(valores: pd.Series, columna_id: str, columna_valor: str) -> pd.DataFrame:
    """Dimension pequena (catalogo) a partir de los valores distintos de
    una columna. La llave se asigna por orden alfabetico, asi es
    determinista: el mismo conjunto de valores produce siempre las mismas
    llaves, particion tras particion."""
    distintos = sorted(valores.dropna().astype(str).unique())
    return pd.DataFrame({columna_id: range(1, len(distintos) + 1), columna_valor: distintos})


# ---------------------------------------------------------------------------
# Dimensiones
# ---------------------------------------------------------------------------
def construir_dim_fecha(fecha: str) -> pd.DataFrame:
    ts = pd.Timestamp(fecha.replace("/", "-"))
    return pd.DataFrame(
        [
            {
                "fecha_id": _fecha_id(fecha),
                "fecha": ts.date(),
                "anio": ts.year,
                "trimestre": ts.quarter,
                "mes": ts.month,
                "dia": ts.day,
                "dia_semana": ts.dayofweek + 1,  # 1 = lunes
                "semana_anio": int(ts.isocalendar().week),
            }
        ]
    )


def construir_dim_cliente(df: pd.DataFrame) -> pd.DataFrame:
    columnas = ["user_id", "age", "gender", "employment_status"]
    columnas += [c for c in ATRIBUTOS_CLIENTE_OPCIONALES if c in df.columns]
    dim = df[columnas].drop_duplicates(subset=["user_id"]).copy()
    dim.insert(2, "rango_edad", dim["age"].apply(_rango_edad))
    return dim.reset_index(drop=True)


def construir_dim_region(df: pd.DataFrame) -> pd.DataFrame:
    return _dimension_catalogo(df["region"], "region_id", "region")


def construir_dim_banda_credito() -> pd.DataFrame:
    """No depende de los datos: las bandas son una regla de negocio fija
    (gold_schema.yml). Se incluye la banda 0 para scores fuera de rango,
    asi ninguna fila de hechos queda con una FK huerfana."""
    filas = [BANDA_SIN_CLASIFICAR] + BANDAS_CREDIT_SCORE
    return pd.DataFrame(
        [
            {"banda_id": b, "rango_credit_score": etiqueta, "score_min": mn, "score_max": mx}
            for b, mn, mx, etiqueta in filas
        ]
    ).astype({"score_min": "Int64", "score_max": "Int64"})


def construir_dim_tipo_prestamo(df: pd.DataFrame) -> pd.DataFrame:
    """loan_type es nulo cuando has_loan = 'No' (~60% de los casos, ver
    silver_schema.yml). En vez de dejar la FK nula, esos clientes apuntan
    a un miembro explicito 'Sin prestamo'."""
    valores = df["loan_type"].fillna(SIN_PRESTAMO) if "loan_type" in df.columns else pd.Series([SIN_PRESTAMO])
    return _dimension_catalogo(valores, "tipo_prestamo_id", "tipo_prestamo")


# ---------------------------------------------------------------------------
# Tabla de hechos
# ---------------------------------------------------------------------------
def construir_fact_posicion_financiera(
    df_calificado: pd.DataFrame, fecha: str, dims: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Grano: un renglon por cliente por fecha de particion (snapshot).

    `df_calificado` es personal_finance_ml de Silver ya con riesgo_score
    (salida de modelo_riesgo.calificar_clientes)."""
    df = df_calificado.copy()

    region_ids = dict(zip(dims["dim_region"]["region"], dims["dim_region"]["region_id"]))
    tipo_ids = dict(
        zip(dims["dim_tipo_prestamo"]["tipo_prestamo"], dims["dim_tipo_prestamo"]["tipo_prestamo_id"])
    )
    tipo_prestamo = df["loan_type"].fillna(SIN_PRESTAMO) if "loan_type" in df.columns else SIN_PRESTAMO

    ingreso = df["monthly_income_usd"].astype(float)
    gasto = df["monthly_expenses_usd"].astype(float)
    ahorro = df["savings_usd"].astype(float)
    flag_prestamo = tiene_prestamo(df["has_loan"])
    # Si no hay prestamo, el monto expuesto es 0 aunque venga nulo.
    monto_prestamo = df["loan_amount_usd"].astype(float).fillna(0.0).where(flag_prestamo == 1, 0.0)

    fact = pd.DataFrame(
        {
            # --- Llaves (FK a cada dimension) ---
            "fecha_id": _fecha_id(fecha),
            "user_id": df["user_id"],
            "region_id": df["region"].astype(str).map(region_ids).astype("Int64"),
            "banda_id": df["credit_score"].apply(banda_credit_score_id),
            "tipo_prestamo_id": pd.Series(tipo_prestamo, index=df.index).astype(str).map(tipo_ids).astype("Int64"),
            # --- Medidas base (vienen de Silver) ---
            "monthly_income_usd": ingreso,
            "monthly_expenses_usd": gasto,
            "savings_usd": ahorro,
            "tiene_prestamo": flag_prestamo,
            "loan_amount_usd": monto_prestamo,
            "debt_to_income_ratio": df["debt_to_income_ratio"].astype(float),
            "credit_score": df["credit_score"],
            # --- Medidas del modelo de riesgo ---
            "riesgo_score": df["riesgo_score"].astype(float),
            "es_riesgo_alto": (df["riesgo_flag"] == "Alto").astype(int),
        }
    )

    for columna in MEDIDAS_OPCIONALES:
        if columna in df.columns:
            fact[columna] = df[columna].astype(float)

    # --- Medidas derivadas a nivel cliente (ver gold_schema.yml,
    # seccion metricas_financieras). Se calculan aqui, una vez, para que
    # cualquier consumidor (KPIs, dashboard, BI) use la misma formula. ---
    ingreso_seguro = ingreso.where(ingreso > 0)  # evita dividir entre 0
    gasto_seguro = gasto.where(gasto > 0)
    fact["flujo_libre_mensual_usd"] = (ingreso - gasto).round(2)
    fact["tasa_ahorro"] = ((ingreso - gasto) / ingreso_seguro).round(4)
    fact["ratio_gasto_ingreso"] = (gasto / ingreso_seguro).round(4)
    fact["meses_cobertura_ahorro"] = (ahorro / gasto_seguro).round(2)
    # Perdida esperada (Basilea): PD x EAD x LGD
    fact["perdida_esperada_usd"] = (fact["riesgo_score"] * monto_prestamo * LGD).round(2)

    return fact


def construir_modelo_estrella(df_calificado: pd.DataFrame, fecha: str) -> dict[str, pd.DataFrame]:
    """Punto de entrada: regresa {nombre_tabla: DataFrame} con las 5
    dimensiones y la tabla de hechos."""
    dims = {
        "dim_fecha": construir_dim_fecha(fecha),
        "dim_cliente": construir_dim_cliente(df_calificado),
        "dim_region": construir_dim_region(df_calificado),
        "dim_banda_credito": construir_dim_banda_credito(),
        "dim_tipo_prestamo": construir_dim_tipo_prestamo(df_calificado),
    }
    fact = construir_fact_posicion_financiera(df_calificado, fecha, dims)
    return {**dims, "fact_posicion_financiera": fact}


# ---------------------------------------------------------------------------
# Integridad del modelo (equivalente en Gold a las expectativas de Silver)
# ---------------------------------------------------------------------------
LLAVES_PRIMARIAS = {
    "dim_fecha": "fecha_id",
    "dim_cliente": "user_id",
    "dim_region": "region_id",
    "dim_banda_credito": "banda_id",
    "dim_tipo_prestamo": "tipo_prestamo_id",
}

RELACIONES = [
    # (columna FK en fact, dimension, columna PK)
    ("fecha_id", "dim_fecha", "fecha_id"),
    ("user_id", "dim_cliente", "user_id"),
    ("region_id", "dim_region", "region_id"),
    ("banda_id", "dim_banda_credito", "banda_id"),
    ("tipo_prestamo_id", "dim_tipo_prestamo", "tipo_prestamo_id"),
]


def validar_integridad(tablas: dict[str, pd.DataFrame]) -> dict:
    """Revisa las dos reglas basicas de un esquema estrella:
      1) Cada PK de dimension es unica y no nula.
      2) Cada FK de la tabla de hechos existe en su dimension (sin huerfanos).
    Ademas el grano del hecho (user_id + fecha_id) debe ser unico."""
    fact = tablas["fact_posicion_financiera"]
    chequeos = []

    for tabla, pk in LLAVES_PRIMARIAS.items():
        serie = tablas[tabla][pk]
        chequeos.append(
            {
                "chequeo": f"pk_unica:{tabla}.{pk}",
                "exito": bool(serie.notna().all() and serie.is_unique),
            }
        )

    for fk, dim, pk in RELACIONES:
        huerfanos = int((~fact[fk].isin(tablas[dim][pk])).sum())
        chequeos.append({"chequeo": f"fk:{fk}->{dim}.{pk}", "exito": huerfanos == 0, "huerfanos": huerfanos})

    duplicados_grano = int(fact.duplicated(subset=["user_id", "fecha_id"]).sum())
    chequeos.append(
        {"chequeo": "grano_unico:fact(user_id,fecha_id)", "exito": duplicados_grano == 0, "duplicados": duplicados_grano}
    )

    return {
        "exito_global": all(c["exito"] for c in chequeos),
        "chequeos_totales": len(chequeos),
        "chequeos_exitosos": sum(c["exito"] for c in chequeos),
        "detalle": chequeos,
    }
