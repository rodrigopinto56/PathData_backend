"""
src/spark/bronze_to_silver.py

Job de limpieza y normalizacion Bronze -> Silver. Aplica las reglas de
calidad ya documentadas en config/data_catalog.yaml (Semana 5-8), separando
registros validos de invalidos (cuarentena) en vez de simplemente
descartarlos -- tal como pide la propuesta ("regla automatizada de
cuarentena que desvia los registros invalidos").

Uso (via spark-submit, orquestado por Airflow):
    spark-submit bronze_to_silver.py --fuente <nombre> --entrada <parquet> --salida <dir>
"""

import argparse
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def clean_loan_default_risk(df):
    """Reglas (config/data_catalog.yaml -> loan_default):
      - Loan_Default_Risk en {0,1}, sin nulos (es el target del modelo)
      - Retirement_Age > 0
      - Debt_Amount >= 0 (si es nulo -> cuarentena)
      - Monthly_Savings >= 0 y no nulo
    """
    df = df.withColumn("Loan_Default_Risk", F.col("Loan_Default_Risk").cast("integer"))
    df = df.withColumn("Retirement_Age", F.col("Retirement_Age").cast("double"))
    df = df.withColumn("Debt_Amount", F.col("Debt_Amount").cast("double"))
    df = df.withColumn("Monthly_Savings", F.col("Monthly_Savings").cast("double"))

    condition_valid = (
        F.col("Loan_Default_Risk").isin(0, 1)
        & (F.col("Retirement_Age").isNotNull() & (F.col("Retirement_Age") > 0))
        & (F.col("Debt_Amount").isNotNull() & (F.col("Debt_Amount") >= 0))
        & (F.col("Monthly_Savings").isNotNull() & (F.col("Monthly_Savings") >= 0))
    )

    df = df.withColumn(
        "motivo_cuarentena",
        F.when(condition_valid, F.lit(None))
        .otherwise(F.lit("Regla de calidad loan default risk fallida"))
    )
    return df, condition_valid


def clean_personal_finance_ml(df):
    """Reglas (config/data_catalog.yaml -> personal_finance_ml):
      - monthly_income_usd >= 0
      - loan_amount_usd >= 0
      - debt_to_income_ratio >= 0
      - credit_score >= 300 and <= 850
    """
    df = df.withColumn("monthly_income_usd", F.col("monthly_income_usd").cast("double"))
    inconsistencia_prestamo = (
        ((F.col("has_loan") == "No") & F.col("loan_type").isNotNull())
        | ((F.col("has_loan") == "Yes") & F.col("loan_type").isNull())
    )

    condition_valid = (
        F.col("user_id").isNotNull()
        & F.col("monthly_income_usd").isNotNull() & (F.col("monthly_income_usd") >= 0)
        & (~inconsistencia_prestamo)
    )

    df = df.withColumn(
        "motivo_cuarentena",
        F.when(condition_valid, F.lit(None))
         .otherwise(F.lit("Inconsistencia has_loan/loan_type o ingreso invalido")),
    )
    return df, condition_valid

CLEANERS = {
    "loan_default_risk": clean_loan_default_risk,
    "personal_finance_ml": clean_personal_finance_ml
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fuente", required=True)
    parser.add_argument("--entrada", required=True)
    parser.add_argument("--salida", required=True)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("bronze_to_silver{args.fuente}").getOrCreate()

    df = spark.read.parquet(args.entrada)
    open_rows = df.count()

    cleaner = CLEANERS.get(args.fuente)
    if cleaner is None:
        raise ValueError(f"No se encontro un limpiador para la fuente {args.fuente}")

    df_marcado, condition_valid = cleaner(df)
    filas_antes_dedup = df_marcado.count()

    if args.fuente == "personal_finance_ml":
        df_marcado = df_marcado.dropDuplicates(["user_id"])
    else:
        df_marcado = df_marcado.dropDuplicates()

    filas_despues_dedup = df_marcado.count()
    duplicados_removidos = filas_antes_dedup - filas_despues_dedup

    print(
        f"[{args.fuente}] dedup: filas_antes={filas_antes_dedup} "
        f"filas_despues={filas_despues_dedup} duplicados_removidos={duplicados_removidos}"
    )

    df_validos = df_marcado.filter(condition_valid).drop("motivo_cuarentena")
    df_cuarentena = df_marcado.filter(~condition_valid)

    validated_rows = df_validos.count()
    cuarentena_rows = df_cuarentena.count()

    df_validos.coalesce(1).write.mode("overwrite").parquet(f"{args.salida}/valid")
    df_cuarentena.coalesce(1).write.mode("overwrite").parquet(f"{args.salida}/quarantine")

    print(
        f"[{args.fuente}] entrada={open_rows} "
        f"validos={validated_rows} cuarentena={cuarentena_rows}"
    )

    spark.stop()

if __name__ == "__main__":
    main()