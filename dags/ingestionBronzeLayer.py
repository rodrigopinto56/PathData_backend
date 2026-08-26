"""
Ingesta los datasets ya descargados por scripts/download_raw_data.py hacia
la capa Bronze, particionada por fecha, con metadata de linaje EMBEBIDA
fisicamente en cada particion
"""

from __future__ import annotations

import json
import os

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException


RAW_DIR = "/opt/airflow/data/raw"
MANIFEST_PATH = f"{RAW_DIR}/manifest.json"
TMP_DIR = "/opt/airflow/data/tmp"

# Backend de almacenamiento configurable 
BRONZE_BACKEND = os.getenv("BRONZE_BACKEND", "local")
BRONZE_PATH = (
    "/opt/airflow/data/bronze" if BRONZE_BACKEND == "local"
    else "s3://bronze"  # requiere conexion S3/MinIO configurada en Airflow
)

# Esquema confirmado en el Diccionario de Datos (Semana 5)
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
        """Lee el CSV, valida columnas minimas y recupera el checksum del
        manifiesto (Paso 1) para propagar linaje verificable hacia Bronze.
        """
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
        """Persiste el dato en Bronze, particionado por fecha, y escribe
        un archivo de metadata junto al parquet: trazabilidad FISICA del
        linaje, no solo una promesa documentada en el catalogo.
        """
        fecha_ingesta = pendulum.now("America/Mexico_City")
        particion = fecha_ingesta.format("YYYY/MM/DD")
        destino = f"{BRONZE_PATH}/{fuente}/{particion}"
        os.makedirs(destino, exist_ok=True)

        df = pd.read_parquet(info["temp_path"])
        df.to_parquet(f"{destino}/data.parquet", index=False)

        metadata = {
            "fuente": fuente,
            "kaggle_id": info["kaggle_id"],
            "sha256_origen": info["sha256_origen"],
            "filas_ingeridas": info["filas"],
            "fecha_ingesta": fecha_ingesta.isoformat(),
        }
        with open(f"{destino}/_ingestion_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    for fuente in RAW_SOURCES:
        info = extraer_y_validar(fuente)
        cargar_a_bronze(fuente, info)


ingesta_bronze_dag()
