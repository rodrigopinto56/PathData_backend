"""
scripts/construir_gold.py

Semana 11, Paso 3: construccion de las tablas Gold.

Por que pandas y no Spark aqui (a diferencia de bronze_to_silver.py):
personal_finance_ml son ~32K filas. Levantar un job de Spark para esto
es anadir un contenedor y un SparkSubmitOperator sin necesidad real de
procesamiento distribuido. Spark se justifica cuando el volumen de datos
lo requiere; a esta escala, pandas es mas simple y mas rapido de correr.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pandas as pd
import s3fs

from modelo_riesgo import STORAGE_OPTIONS, MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET
from modelo_riesgo import SILVER_BUCKET, entrenar_modelo, calificar_clientes

import os

GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold-layer")

# Ver config/gold_schema.yml, seccion tablas.gold_clientes_riesgo
COLUMNAS_CLIENTES_RIESGO = [
    "user_id", "age", "gender", "region", "employment_status",
    "monthly_income_usd", "monthly_expenses_usd", "savings_usd",
    "has_loan", "loan_amount_usd", "debt_to_income_ratio", "credit_score",
    "riesgo_score", "riesgo_flag",
]

# Ver config/gold_schema.yml, seccion bandas_credit_score
BANDAS_CREDIT_SCORE = [
    (300, 579, "Bajo"),
    (580, 669, "Regular"),
    (670, 739, "Bueno"),
    (740, 799, "Muy bueno"),
    (800, 850, "Excelente"),
]


def _bucket_existe_o_crear(fs: s3fs.S3FileSystem, bucket: str) -> None:
    """El bucket gold-layer no se crea a mano en la consola de MinIO como
    bronze-layer/silver-layer -- se crea solo la primera vez que corre
    este script, para no depender de un paso manual mas."""
    if not fs.exists(bucket):
        fs.mkdir(bucket)


def _asignar_banda_credit_score(score: int) -> str:
    for minimo, maximo, etiqueta in BANDAS_CREDIT_SCORE:
        if minimo <= score <= maximo:
            return etiqueta
    return "Sin clasificar"


def construir_gold_particion(fecha: str) -> dict:
    """Construye las dos tablas Gold para `fecha` y las sube a MinIO.
    Regresa un resumen (conteos + metricas del modelo) para logging."""
    fs = s3fs.S3FileSystem(
        key=MINIO_KEY, secret=MINIO_SECRET,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )
    _bucket_existe_o_crear(fs, GOLD_BUCKET)

    # --- Entrenar + calificar ---
    resultado_entrenamiento = entrenar_modelo(fecha)
    modelo = resultado_entrenamiento["modelo"]
    metricas_modelo = resultado_entrenamiento["metricas"]

    df_pf = pd.read_parquet(
        f"s3://{SILVER_BUCKET}/personal_finance_ml/{fecha}/valid",
        storage_options=STORAGE_OPTIONS,
    )
    df_calificado = calificar_clientes(df_pf, modelo)

    # --- Tabla 1: gold_clientes_riesgo (detalle por cliente) ---
    df_clientes = df_calificado[COLUMNAS_CLIENTES_RIESGO].copy()
    ruta_clientes = f"s3://{GOLD_BUCKET}/gold_clientes_riesgo/{fecha}/data.parquet"
    df_clientes.to_parquet(ruta_clientes, storage_options=STORAGE_OPTIONS, index=False)

    # --- Tabla 2: gold_metricas_por_segmento (agregado por region x banda) ---
    df_calificado["rango_credit_score"] = df_calificado["credit_score"].apply(
        _asignar_banda_credit_score
    )

    agregado = (
        df_calificado.groupby(["region", "rango_credit_score"])
        .agg(
            num_clientes=("user_id", "count"),
            ingreso_promedio_usd=("monthly_income_usd", "mean"),
            ahorro_promedio_usd=("savings_usd", "mean"),
            deuda_ingreso_promedio=("debt_to_income_ratio", "mean"),
        )
        .reset_index()
    )

    pct_prestamo = (
        df_calificado.groupby(["region", "rango_credit_score"])["has_loan"]
        .apply(lambda s: round((s == "Yes").mean() * 100, 1))
        .reset_index(name="pct_con_prestamo")
    )
    pct_riesgo = (
        df_calificado.groupby(["region", "rango_credit_score"])["riesgo_flag"]
        .apply(lambda s: round((s == "Alto").mean() * 100, 1))
        .reset_index(name="pct_riesgo_alto")
    )

    df_segmento = agregado.merge(
        pct_prestamo, on=["region", "rango_credit_score"]
    ).merge(pct_riesgo, on=["region", "rango_credit_score"])

    columnas_redondeo = ["ingreso_promedio_usd", "ahorro_promedio_usd", "deuda_ingreso_promedio"]
    df_segmento[columnas_redondeo] = df_segmento[columnas_redondeo].round(2)

    ruta_segmento = f"s3://{GOLD_BUCKET}/gold_metricas_por_segmento/{fecha}/data.parquet"
    df_segmento.to_parquet(ruta_segmento, storage_options=STORAGE_OPTIONS, index=False)

    # --- Metadata (mismo patron que _silver_metadata.json de la Semana 11 pasada) ---
    resumen = {
        "fecha": fecha,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "modelo": metricas_modelo,
        "gold_clientes_riesgo": {"filas": int(len(df_clientes))},
        "gold_metricas_por_segmento": {"filas": int(len(df_segmento))},
    }
    ruta_metadata = f"s3://{GOLD_BUCKET}/_gold_metadata/{fecha.replace('/', '-')}.json"
    with fs.open(ruta_metadata, "w") as f:
        json.dump(resumen, f, indent=2, default=str)

    return resumen


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    resumen = construir_gold_particion(fecha)
    print(json.dumps(resumen, indent=2, default=str))


if __name__ == "__main__":
    main()