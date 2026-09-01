"""
dags/ingestionBronzeLayer.py  (v4 - soporte real para MinIO via S3Hook)

Ingesta loan_default_risk y personal_finance_ml hacia Bronze, particionado
por la fecha LOGICA del DAG run. Soporta 2 backends intercambiables via la
variable de entorno BRONZE_BACKEND:
  - "local": escribe en disco (volumen Docker), como hasta ahora.
  - "minio": escribe al bucket de MinIO usando S3Hook (lee credenciales
    de la Connection 'minio_s3_conn', nunca hardcodeadas en el codigo).
"""

from __future__ import annotations
import io
import json
import os
import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.operators.python import get_current_context

RAW_DIR = "/opt/airflow/data/raw"
MANIFEST_PATH = f"{RAW_DIR}/manifest.json"
TMP_DIR = "/opt/airflow/data/tmp"
BRONZE_BACKEND = os.getenv("BRONZE_BACKEND", "local")
BRONZE_PATH_LOCAL = "/opt/airflow/data/bronze"
BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze-layer")  
S3_CONN_ID = os.getenv("BRONZE_S3_CONN_ID", "minio_s3_conn")

RAW_SOURCES = {
    "loan_default_risk": {
        "path": f"{RAW_DIR}/loan_default_risk_dataset.csv",
        "expected_columns": [
            "Retirement_Age", "Debt_Amount", "Monthly_Savings", "Loan_Default_Risk",
        ],
    },
    "personal_finance_ml": {
        "path": f"{RAW_DIR}/synthetic_personal_finance_dataset.csv",
        "expected_columns": [
            "user_id", "monthly_income_usd", "has_loan", "loan_type",
            "loan_amount_usd", "debt_to_income_ratio", "credit_score",
        ],
    },
}

default_args = {
    "owner": "data-engineering",
    "retries": 3,
    "retry_delay": pendulum.duration(minutes=5),
}

@dag(
    dag_id="ingesta_bronze_dag",
    description="Ingesta loan_default_risk y personal_finance_ml hacia Bronze",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Mexico_City"),
    catchup=False,
    default_args=default_args,
    tags=["bronze", "ingesta", "gemelo-digital-financiero"],
)
def ingesta_bronze_dag():

    @task
    def extraer_y_validar(fuente: str) -> dict:
        config = RAW_SOURCES[fuente]
        df = pd.read_csv(config["path"])

        faltantes = [c for c in config["expected_columns"] if c not in df.columns]
        if faltantes:
            raise AirflowFailException(
                f"Faltan columnas esperadas en '{fuente}': {faltantes}"
            )

        with open(MANIFEST_PATH) as f:
            manifiesto = json.load(f)
        nombre_archivo = os.path.basename(config["path"])
        info_manifiesto = manifiesto.get(nombre_archivo, {})

        os.makedirs(TMP_DIR, exist_ok=True)
        temp_path = f"{TMP_DIR}/{fuente}.parquet"
        df.to_parquet(temp_path, index=False)

        return {
            "temp_path": temp_path,
            "filas": len(df),
            "sha256_origen": info_manifiesto.get("sha256", "desconocido"),
            "kaggle_id": info_manifiesto.get("kaggle_id", "desconocido"),
        }

    @task
    def cargar_a_bronze(fuente: str, info: dict) -> None:
        """Persiste el dato en Bronze, particionado por la fecha LOGICA del
        DAG run. El destino fisico (disco local o MinIO) se decide por
        BRONZE_BACKEND, sin cambiar la logica de particionamiento.
        """
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        df = pd.read_parquet(info["temp_path"])

        metadata = {
            "fuente": fuente,
            "kaggle_id": info["kaggle_id"],
            "sha256_origen": info["sha256_origen"],
            "filas_ingeridas": info["filas"],
            "fecha_logica_particion": fecha_logica.isoformat(),
            "fecha_ejecucion_real": pendulum.now("America/Mexico_City").isoformat(),
            "backend": BRONZE_BACKEND,
        }

        if BRONZE_BACKEND == "minio":
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook
            hook = S3Hook(aws_conn_id=S3_CONN_ID)
            buffer_parquet = io.BytesIO()
            df.to_parquet(buffer_parquet, index=False)
            buffer_parquet.seek(0)
            key_parquet = f"{fuente}/{particion}/data.parquet"
            hook.load_bytes(
                buffer_parquet.read(),
                key=key_parquet,
                bucket_name=BRONZE_BUCKET,
                replace=True,
            )

            key_meta = f"{fuente}/{particion}/_ingestion_metadata.json"
            hook.load_string(
                json.dumps(metadata, indent=2),
                key=key_meta,
                bucket_name=BRONZE_BUCKET,
                replace=True,
            )
        else:
            destino = f"{BRONZE_PATH_LOCAL}/{fuente}/{particion}"
            os.makedirs(destino, exist_ok=True)
            df.to_parquet(f"{destino}/data.parquet", index=False)
            with open(f"{destino}/_ingestion_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

    for fuente in RAW_SOURCES:
        info = extraer_y_validar(fuente)
        cargar_a_bronze(fuente, info)


ingesta_bronze_dag()