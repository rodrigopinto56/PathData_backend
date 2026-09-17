"""
dags/transformacion_silver_dag.py

Orquesta Bronze -> Silver:
  1) Descarga la particion Bronze del dia (MinIO) a un staging local
     compartido con el cluster Spark.
  2) Somete el job de limpieza (bronze_to_silver.py) via SparkSubmitOperator
     -- driver en Airflow, executors en el cluster Spark separado.
  3) Sube el resultado (valid + quarantine) de vuelta a MinIO, bucket
     silver-layer.
  4) Valida la calidad de la particion recien subida con Great
     Expectations (Semana 11), a partir del contrato config/silver_schema.yml.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pendulum

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

sys.path.append("/opt/airflow/scripts")
from validar_calidad_silver import validar_particion


logger = logging.getLogger("transformacion_silver_dag")

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze-layer")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver-layer")
S3_CONN_ID = os.getenv("BRONZE_S3_CONN_ID", "minio_s3_conn")
STAGING_DIR = "/opt/airflow/data/tmp/silver_staging"
LOG_EVENTS_PATH = "/opt/airflow/data/reports/pipeline_log.jsonl"

FUENTES = ["loan_default_risk", "personal_finance_ml"]

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
}


def registrar_evento(evento: dict) -> None:
    """Escribe un evento estructurado (JSON Lines) para observabilidad.
    Comparte el mismo archivo de log que ingesta_bronze_dag.py, para tener
    una sola fuente de eventos de toda la malla de datos (util para el
    dashboard de calidad de la Semana 12).
    """
    evento["timestamp"] = pendulum.now("America/Mexico_City").isoformat()
    os.makedirs(os.path.dirname(LOG_EVENTS_PATH), exist_ok=True)
    with open(LOG_EVENTS_PATH, "a") as f:
        f.write(json.dumps(evento) + "\n")
    logger.info(json.dumps(evento))


@dag(
    dag_id="transformacion_silver_dag",
    description="Transforma Bronze -> Silver (limpieza, normalizacion, cuarentena, calidad)",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Mexico_City"),
    catchup=False,
    default_args=default_args,
    tags=["silver", "pyspark", "gemelo-digital-financiero"],
)
def transformacion_silver_dag():

    @task
    def descargar_particion_bronze(fuente: str) -> str:
        """Descarga la particion Bronze de la fecha logica actual desde
        MinIO hacia un staging local que Spark puede leer via volumen
        compartido."""
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        hook = S3Hook(aws_conn_id=S3_CONN_ID)
        key = f"{fuente}/{particion}/data.parquet"

        destino_local = f"{STAGING_DIR}/{fuente}/{particion}/data.parquet"
        os.makedirs(os.path.dirname(destino_local), exist_ok=True)

        obj = hook.get_key(key, bucket_name=BRONZE_BUCKET)
        obj.download_file(destino_local)

        return destino_local

    @task
    def subir_silver_a_minio(fuente: str, ruta_salida_local: str) -> dict:
        """Sube valid/ y quarantine/ (generados por Spark en el volumen
        local) de vuelta a MinIO, bucket silver-layer.
        """
        hook = S3Hook(aws_conn_id=S3_CONN_ID)
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        subidos = {}
        for subcarpeta in ["valid", "quarantine"]:
            local_path = f"{ruta_salida_local}/{subcarpeta}"
            if not os.path.isdir(local_path):
                continue

            prefijo = f"{fuente}/{particion}/{subcarpeta}/"
            claves_existentes = hook.list_keys(bucket_name=SILVER_BUCKET, prefix=prefijo)
            if claves_existentes:
                hook.delete_objects(bucket=SILVER_BUCKET, keys=claves_existentes)

            archivos_subidos = 0
            for nombre_archivo in os.listdir(local_path):
                if nombre_archivo.startswith("."):
                    continue
                local_file = f"{local_path}/{nombre_archivo}"
                key = f"{prefijo}{nombre_archivo}"
                hook.load_file(local_file, key=key, bucket_name=SILVER_BUCKET, replace=True)
                archivos_subidos += 1
            subidos[subcarpeta] = archivos_subidos

        return subidos

    @task
    def validar_calidad(fuente: str) -> dict:
        """Corre el Expectation Suite (Great Expectations) contra la
        particion recien subida a Silver. No bloquea el pipeline si falla
        -- por ahora solo reporta, para no arriesgar el resto de la malla
        mientras el framework de calidad esta en su primera iteracion."""
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        resultado = validar_particion(fuente, particion)

        registrar_evento({
            "nivel": "INFO" if resultado["exito_global"] else "ALERTA",
            "task": "validar_calidad",
            "fuente": fuente,
            "particion": particion,
            "exito_global": resultado["exito_global"],
            "expectativas_exitosas": resultado["expectativas_exitosas"],
            "expectativas_totales": resultado["expectativas_totales"],
        })
         
        hook = S3Hook(aws_conn_id=S3_CONN_ID)
        key = f"{fuente}/{particion}/_calidad_reporte.json"
        hook.load_string(
            json.dumps(resultado, indent=2, default=str),
            key=key,
            bucket_name=SILVER_BUCKET,
            replace=True,
        )

        return resultado
    
    @task
    def congelar_silver_v1(fuente: str, resultado_calidad: dict) -> None:
        """Escribe el metadata de cierre de 'Dataset Silver v1' (Semana 10):
        version formal, fecha de congelamiento y resultado de calidad,
        junto a valid/ y quarantine/ en MinIO. Equivalente en Silver al
        _ingestion_metadata.json que ya existe en Bronze."""
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        metadata = {
            "version_dataset": "silver_v1",
            "fuente": fuente,
            "particion": particion,
            "congelado_en": pendulum.now("America/Mexico_City").isoformat(),
            "calidad": {
                "exito_global": resultado_calidad["exito_global"],
                "expectativas_exitosas": resultado_calidad["expectativas_exitosas"],
                "expectativas_totales": resultado_calidad["expectativas_totales"],
                "filas_evaluadas": resultado_calidad["filas_evaluadas"],
            },
        }

        hook = S3Hook(aws_conn_id=S3_CONN_ID)
        key = f"{fuente}/{particion}/_silver_metadata.json"
        hook.load_string(
            json.dumps(metadata, indent=2, default=str),
            key=key,
            bucket_name=SILVER_BUCKET,
            replace=True,
        )

    for fuente in FUENTES:
        ruta_bronze_local = descargar_particion_bronze(fuente)
        salida_silver = f"{STAGING_DIR}/{fuente}/silver_output"

        limpiar_y_normalizar = SparkSubmitOperator(
            task_id=f"limpiar_{fuente}",
            application="/opt/spark-apps/bronze_to_silver.py",
            conn_id="spark_default",
            application_args=[
                "--fuente", fuente,
                "--entrada", ruta_bronze_local,
                "--salida", salida_silver,
            ],
        )

        ruta_subida = subir_silver_a_minio(fuente, salida_silver)
        resultado_calidad = validar_calidad(fuente)
        congelar_silver_v1(fuente, resultado_calidad)  # <-- esta línea faltaba

        ruta_bronze_local >> limpiar_y_normalizar >> ruta_subida >> resultado_calidad


transformacion_silver_dag()