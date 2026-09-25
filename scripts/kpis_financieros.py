"""
scripts/kpis_financieros.py

Semana 12: KPIs financieros sobre el modelo estrella de Gold.

Tres familias de KPIs (ver config/gold_schema.yml, seccion
metricas_financieras, para formula, unidad y umbral de cada uno):

  1) Riesgo financiero        -> que tan probable es que la cartera incumpla
  2) Exposicion               -> cuanto dinero esta en juego y donde se concentra
  3) Comportamiento transaccional -> como ingresan, gastan y ahorran los clientes

Todos se calculan a partir de fact_posicion_financiera + dimensiones
(nunca directo de Silver), asi que una sola definicion de cada medida
alimenta tanto los KPIs como el dashboard.

Salidas (gold-layer, particionadas por fecha):
  - gold_kpis_financieros:   formato largo, un renglon por KPI. Incluye
                             umbral y estado (OK / ALERTA) para que el
                             dashboard pueda senalar que requiere accion.
  - gold_kpis_por_segmento:  los KPIs clave abiertos por region, banda de
                             credit score, tipo de prestamo, rango de edad,
                             situacion laboral y perfil de ahorro.

Igual que modelo_dimensional.py, las funciones de calculo son pandas
puro; solo calcular_kpis_particion() toca MinIO.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from modelo_dimensional import LGD

GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold-layer")

# ---------------------------------------------------------------------------
# Parametros de negocio (ver config/gold_schema.yml, parametros_financieros)
# ---------------------------------------------------------------------------
# DTI > 43% es el limite que usa el CFPB (EE.UU.) para una hipoteca
# "calificada"; es la referencia mas comun para decir "sobreendeudado".
# Supone que debt_to_income_ratio viene como fraccion (0.43 = 43%).
UMBRAL_DTI_ALTO = 0.43
# Menos de 3 meses de gastos ahorrados = sin fondo de emergencia.
MESES_COLCHON_MINIMO = 3
# Perfil de ahorro segun tasa de ahorro mensual (regla 50/30/20: ahorrar 20%).
UMBRAL_AHORRADOR = 0.20

# Catalogo de KPIs: (categoria, unidad, descripcion, umbral, alerta_si)
#   alerta_si = "mayor"  -> ALERTA cuando valor > umbral
#   alerta_si = "menor"  -> ALERTA cuando valor < umbral
#   umbral = None        -> KPI informativo, sin semaforo
RIESGO = "riesgo_financiero"
EXPOSICION = "exposicion"
COMPORTAMIENTO = "comportamiento_transaccional"

CATALOGO_KPIS = {
    # --- Riesgo financiero ---
    "tasa_incumplimiento_historica_pct": (RIESGO, "%", "% de incumplimiento observado en loan_default_risk (dato real, no modelado)", None, None),
    "probabilidad_incumplimiento_promedio_pct": (RIESGO, "%", "PD promedio de la cartera segun el modelo de riesgo", 30.0, "mayor"),
    "pct_clientes_riesgo_alto": (RIESGO, "%", "% de clientes con riesgo_flag = Alto (PD >= 0.5)", 20.0, "mayor"),
    "credit_score_promedio": (RIESGO, "puntos", "Credit score promedio de la cartera", 670.0, "menor"),
    "pct_clientes_subprime": (RIESGO, "%", "% de clientes con credit score < 580 (banda 'Bajo')", 20.0, "mayor"),
    "dti_mediana": (RIESGO, "ratio", "Mediana de debt_to_income_ratio entre clientes con prestamo", None, None),
    "pct_clientes_sobreendeudados": (RIESGO, "%", f"% de clientes con prestamo y DTI > {UMBRAL_DTI_ALTO}", 25.0, "mayor"),
    "auc_modelo_riesgo": (RIESGO, "ratio", "Poder discriminante del modelo que genera la PD (0.5 = azar, 1 = perfecto)", 0.70, "menor"),
    # --- Exposicion ---
    "num_prestamos_activos": (EXPOSICION, "prestamos", "Clientes con prestamo vigente", None, None),
    "pct_clientes_con_prestamo": (EXPOSICION, "%", "% de clientes con prestamo vigente", None, None),
    "exposicion_total_usd": (EXPOSICION, "USD", "Suma de loan_amount_usd (EAD) de la cartera", None, None),
    "exposicion_promedio_usd": (EXPOSICION, "USD", "Monto promedio por prestamo", None, None),
    "exposicion_riesgo_alto_usd": (EXPOSICION, "USD", "Monto prestado a clientes de riesgo alto", None, None),
    "pct_exposicion_riesgo_alto": (EXPOSICION, "%", "% de la exposicion total concentrada en clientes de riesgo alto", 20.0, "mayor"),
    "perdida_esperada_usd": (EXPOSICION, "USD", f"Perdida esperada = sum(PD x EAD x LGD), LGD = {LGD:.0%}", None, None),
    "perdida_esperada_pct_exposicion": (EXPOSICION, "%", "Perdida esperada como % de la exposicion total", 5.0, "mayor"),
    "indice_concentracion_region_hhi": (EXPOSICION, "indice 0-1", "Herfindahl-Hirschman de la exposicion por region (>0.25 = concentrada)", 0.25, "mayor"),
    "pct_exposicion_region_principal": (EXPOSICION, "%", "% de la exposicion en la region con mas saldo", None, None),
    # --- Comportamiento transaccional ---
    "ingreso_mensual_promedio_usd": (COMPORTAMIENTO, "USD", "Promedio de monthly_income_usd", None, None),
    "gasto_mensual_promedio_usd": (COMPORTAMIENTO, "USD", "Promedio de monthly_expenses_usd", None, None),
    "ratio_gasto_ingreso_mediana": (COMPORTAMIENTO, "ratio", "Mediana de gasto / ingreso mensual", None, None),
    "tasa_ahorro_mediana_pct": (COMPORTAMIENTO, "%", "Mediana de (ingreso - gasto) / ingreso", UMBRAL_AHORRADOR * 100, "menor"),
    "flujo_libre_mensual_mediana_usd": (COMPORTAMIENTO, "USD", "Mediana de ingreso - gasto mensual", None, None),
    "pct_clientes_deficitarios": (COMPORTAMIENTO, "%", "% de clientes que gastan mas de lo que ingresan", 15.0, "mayor"),
    "meses_cobertura_ahorro_mediana": (COMPORTAMIENTO, "meses", "Mediana de meses de gasto cubiertos con el ahorro", float(MESES_COLCHON_MINIMO), "menor"),
    "pct_clientes_sin_colchon": (COMPORTAMIENTO, "%", f"% de clientes con menos de {MESES_COLCHON_MINIMO} meses de gasto ahorrados", 30.0, "mayor"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(numerador: float, denominador: float) -> float | None:
    return round(numerador / denominador * 100, 2) if denominador else None


def _redondear(valor) -> float | None:
    if valor is None or pd.isna(valor):
        return None
    return round(float(valor), 4)


def perfil_ahorro(tasa_ahorro) -> str:
    if pd.isna(tasa_ahorro):
        return "Sin ingreso"
    if tasa_ahorro < 0:
        return "Deficitario"
    if tasa_ahorro < UMBRAL_AHORRADOR:
        return "Equilibrado"
    return "Ahorrador"


def vista_analitica(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une fact_posicion_financiera con sus dimensiones (el clasico
    'desnormalizar para consultar' de un esquema estrella). Es la base
    de los KPIs segmentados."""
    fact = tablas["fact_posicion_financiera"]
    vista = (
        fact.merge(tablas["dim_region"], on="region_id", how="left")
        .merge(tablas["dim_banda_credito"][["banda_id", "rango_credit_score"]], on="banda_id", how="left")
        .merge(tablas["dim_tipo_prestamo"], on="tipo_prestamo_id", how="left")
        .merge(
            tablas["dim_cliente"][["user_id", "rango_edad", "employment_status"]],
            on="user_id",
            how="left",
        )
    )
    vista["perfil_ahorro"] = vista["tasa_ahorro"].apply(perfil_ahorro)
    return vista


def _hhi(participaciones: pd.Series) -> float | None:
    """Indice Herfindahl-Hirschman: suma de participaciones al cuadrado.
    1/N si todo esta repartido igual entre N grupos, 1.0 si todo esta en uno."""
    total = participaciones.sum()
    if not total:
        return None
    return float(((participaciones / total) ** 2).sum())


# ---------------------------------------------------------------------------
# KPIs globales
# ---------------------------------------------------------------------------
def kpis_riesgo(fact: pd.DataFrame, df_loan_default: pd.DataFrame | None, metricas_modelo: dict | None) -> dict:
    n = len(fact)
    con_prestamo = fact[fact["tiene_prestamo"] == 1]
    dti = con_prestamo["debt_to_income_ratio"].dropna()

    tasa_historica = None
    if df_loan_default is not None and "Loan_Default_Risk" in df_loan_default.columns and len(df_loan_default):
        tasa_historica = round(float(df_loan_default["Loan_Default_Risk"].mean()) * 100, 2)

    return {
        "tasa_incumplimiento_historica_pct": tasa_historica,
        "probabilidad_incumplimiento_promedio_pct": round(float(fact["riesgo_score"].mean()) * 100, 2) if n else None,
        "pct_clientes_riesgo_alto": _pct(fact["es_riesgo_alto"].sum(), n),
        "credit_score_promedio": _redondear(fact["credit_score"].mean()),
        "pct_clientes_subprime": _pct((fact["credit_score"] < 580).sum(), n),
        "dti_mediana": _redondear(dti.median()) if len(dti) else None,
        "pct_clientes_sobreendeudados": _pct((dti > UMBRAL_DTI_ALTO).sum(), len(con_prestamo)),
        "auc_modelo_riesgo": (metricas_modelo or {}).get("auc"),
    }


def kpis_exposicion(vista: pd.DataFrame) -> dict:
    n = len(vista)
    con_prestamo = vista[vista["tiene_prestamo"] == 1]
    exposicion_total = float(con_prestamo["loan_amount_usd"].sum())
    exposicion_riesgo_alto = float(con_prestamo.loc[con_prestamo["es_riesgo_alto"] == 1, "loan_amount_usd"].sum())
    perdida_esperada = float(vista["perdida_esperada_usd"].sum())
    por_region = con_prestamo.groupby("region")["loan_amount_usd"].sum()

    return {
        "num_prestamos_activos": int(len(con_prestamo)),
        "pct_clientes_con_prestamo": _pct(len(con_prestamo), n),
        "exposicion_total_usd": round(exposicion_total, 2),
        "exposicion_promedio_usd": round(exposicion_total / len(con_prestamo), 2) if len(con_prestamo) else None,
        "exposicion_riesgo_alto_usd": round(exposicion_riesgo_alto, 2),
        "pct_exposicion_riesgo_alto": _pct(exposicion_riesgo_alto, exposicion_total),
        "perdida_esperada_usd": round(perdida_esperada, 2),
        "perdida_esperada_pct_exposicion": _pct(perdida_esperada, exposicion_total),
        "indice_concentracion_region_hhi": _redondear(_hhi(por_region)),
        "pct_exposicion_region_principal": _pct(por_region.max(), exposicion_total) if len(por_region) else None,
    }


def kpis_comportamiento(fact: pd.DataFrame) -> dict:
    n = len(fact)
    return {
        "ingreso_mensual_promedio_usd": _redondear(fact["monthly_income_usd"].mean()),
        "gasto_mensual_promedio_usd": _redondear(fact["monthly_expenses_usd"].mean()),
        "ratio_gasto_ingreso_mediana": _redondear(fact["ratio_gasto_ingreso"].median()),
        "tasa_ahorro_mediana_pct": _redondear(fact["tasa_ahorro"].median() * 100),
        "flujo_libre_mensual_mediana_usd": _redondear(fact["flujo_libre_mensual_usd"].median()),
        "pct_clientes_deficitarios": _pct((fact["flujo_libre_mensual_usd"] < 0).sum(), n),
        "meses_cobertura_ahorro_mediana": _redondear(fact["meses_cobertura_ahorro"].median()),
        "pct_clientes_sin_colchon": _pct((fact["meses_cobertura_ahorro"] < MESES_COLCHON_MINIMO).sum(), n),
    }


def _estado(valor, umbral, alerta_si) -> str:
    if umbral is None or valor is None:
        return "INFO"
    if alerta_si == "mayor":
        return "ALERTA" if valor > umbral else "OK"
    return "ALERTA" if valor < umbral else "OK"


def calcular_kpis_globales(
    tablas: dict[str, pd.DataFrame],
    fecha: str,
    df_loan_default: pd.DataFrame | None = None,
    metricas_modelo: dict | None = None,
) -> pd.DataFrame:
    """Regresa gold_kpis_financieros: un renglon por KPI (formato largo)."""
    fact = tablas["fact_posicion_financiera"]
    vista = vista_analitica(tablas)

    valores = {
        **kpis_riesgo(fact, df_loan_default, metricas_modelo),
        **kpis_exposicion(vista),
        **kpis_comportamiento(fact),
    }

    filas = []
    for kpi, (categoria, unidad, descripcion, umbral, alerta_si) in CATALOGO_KPIS.items():
        valor = valores.get(kpi)
        filas.append(
            {
                "fecha": fecha,
                "categoria": categoria,
                "kpi": kpi,
                "valor": None if valor is None else float(valor),
                "unidad": unidad,
                "umbral": umbral,
                "estado": _estado(valor, umbral, alerta_si),
                "descripcion": descripcion,
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# KPIs por segmento
# ---------------------------------------------------------------------------
DIMENSIONES_SEGMENTO = [
    "region",
    "rango_credit_score",
    "tipo_prestamo",
    "rango_edad",
    "employment_status",
    "perfil_ahorro",
]


def calcular_kpis_por_segmento(tablas: dict[str, pd.DataFrame], fecha: str) -> pd.DataFrame:
    """Regresa gold_kpis_por_segmento: un renglon por (dimension, segmento)
    con los KPIs clave de las tres familias, para responder "donde" esta
    el riesgo / la exposicion / el mal comportamiento de pago."""
    vista = vista_analitica(tablas)
    vista["exposicion_usd"] = vista["loan_amount_usd"].where(vista["tiene_prestamo"] == 1, 0.0)
    vista["es_deficitario"] = (vista["flujo_libre_mensual_usd"] < 0).astype(int)
    exposicion_total = vista["exposicion_usd"].sum()

    partes = []
    for dimension in DIMENSIONES_SEGMENTO:
        agrupado = (
            vista.groupby(dimension, dropna=False)
            .agg(
                num_clientes=("user_id", "count"),
                pd_promedio_pct=("riesgo_score", "mean"),
                pct_riesgo_alto=("es_riesgo_alto", "mean"),
                credit_score_promedio=("credit_score", "mean"),
                num_prestamos=("tiene_prestamo", "sum"),
                exposicion_usd=("exposicion_usd", "sum"),
                perdida_esperada_usd=("perdida_esperada_usd", "sum"),
                ingreso_promedio_usd=("monthly_income_usd", "mean"),
                tasa_ahorro_mediana_pct=("tasa_ahorro", "median"),
                pct_deficitarios=("es_deficitario", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "segmento"})
        )
        agrupado.insert(0, "dimension", dimension)
        partes.append(agrupado)

    df = pd.concat(partes, ignore_index=True)
    df["segmento"] = df["segmento"].fillna("Sin dato").astype(str)
    for columna in ["pd_promedio_pct", "pct_riesgo_alto", "tasa_ahorro_mediana_pct", "pct_deficitarios"]:
        df[columna] = df[columna] * 100
    df["pct_exposicion_del_total"] = df["exposicion_usd"] / exposicion_total * 100 if exposicion_total else None
    df["perdida_esperada_pct_exposicion"] = (
        (df["perdida_esperada_usd"] / df["exposicion_usd"].where(df["exposicion_usd"] > 0)) * 100
    )
    df.insert(0, "fecha", fecha)
    return df.round(2)


# ---------------------------------------------------------------------------
# E/S con MinIO (lo unico que no es pandas puro)
# ---------------------------------------------------------------------------
TABLAS_ESTRELLA = [
    "dim_fecha",
    "dim_cliente",
    "dim_region",
    "dim_banda_credito",
    "dim_tipo_prestamo",
    "fact_posicion_financiera",
]


def calcular_kpis_particion(fecha: str) -> dict:
    """Lee el modelo estrella de `fecha` desde gold-layer, calcula ambas
    tablas de KPIs y las sube a MinIO. Regresa un resumen para logging."""
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET, SILVER_BUCKET, STORAGE_OPTIONS

    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})

    tablas = {
        nombre: pd.read_parquet(f"s3://{GOLD_BUCKET}/{nombre}/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS)
        for nombre in TABLAS_ESTRELLA
    }

    # Tasa de incumplimiento real (target historico del modelo)
    df_loan_default = pd.read_parquet(
        f"s3://{SILVER_BUCKET}/loan_default_risk/{fecha}/valid", storage_options=STORAGE_OPTIONS
    )

    # Metricas del modelo (AUC) que construir_gold ya dejo en _gold_metadata
    ruta_metadata = f"{GOLD_BUCKET}/_gold_metadata/{fecha.replace('/', '-')}.json"
    metricas_modelo = None
    if fs.exists(ruta_metadata):
        with fs.open(ruta_metadata, "r") as f:
            metricas_modelo = json.load(f).get("modelo")

    df_kpis = calcular_kpis_globales(tablas, fecha, df_loan_default, metricas_modelo)
    df_segmento = calcular_kpis_por_segmento(tablas, fecha)

    df_kpis.to_parquet(
        f"s3://{GOLD_BUCKET}/gold_kpis_financieros/{fecha}/data.parquet",
        storage_options=STORAGE_OPTIONS,
        index=False,
    )
    df_segmento.to_parquet(
        f"s3://{GOLD_BUCKET}/gold_kpis_por_segmento/{fecha}/data.parquet",
        storage_options=STORAGE_OPTIONS,
        index=False,
    )

    alertas = df_kpis[df_kpis["estado"] == "ALERTA"]["kpi"].tolist()
    return {
        "fecha": fecha,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "kpis_calculados": int(df_kpis["valor"].notna().sum()),
        "kpis_totales": int(len(df_kpis)),
        "kpis_en_alerta": alertas,
        "gold_kpis_por_segmento": {"filas": int(len(df_segmento))},
    }


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(calcular_kpis_particion(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
