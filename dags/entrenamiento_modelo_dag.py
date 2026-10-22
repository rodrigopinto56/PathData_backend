"""
dags/entrenamiento_modelo_dag.py

Semanas 14-15: ciclo de vida del modelo de riesgo crediticio, separado de
la calificacion diaria (transformacion_gold_dag solo CARGA el campeon).

  1) construir_dataset_features (S14): credit_risk (Silver) -> features,
     seleccion de variables (IV/WoE), particion y validacion ->
     gold_features_entrenamiento + reporte_seleccion_variables.
  2) entrenar_y_registrar (S15): scorecard vs boosting, validacion cruzada,
     calibracion, pruebas de negocio, equidad -> version nueva en el
     registro (ml-models/riesgo_credito/<version>/) y promocion a campeon
     solo si no empeora al vigente.

Semanal: re-entrenar cada dia no tiene sentido (los datos de entrenamiento
no cambian tan rapido) y deja sin control que modelo esta en produccion.
Tambien se puede disparar a mano desde la UI de Airflow.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pendulum

from airflow.decorators import dag, task
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import get_current_context

sys.path.append("/opt/airflow/scripts")
from features import construir_dataset_entrenamiento
from modelo_credito import entrenar_y_registrar

logger = logging.getLogger("entrenamiento_modelo_dag")

LOG_EVENTS_PATH = "/opt/airflow/data/reports/pipeline_log.jsonl"

default_args = {
    "owner": "data-engineering",
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


def registrar_evento(evento: dict) -> None:
    """Mismo log compartido que el resto de los DAGs."""
    evento["timestamp"] = pendulum.now("America/Mexico_City").isoformat()
    os.makedirs(os.path.dirname(LOG_EVENTS_PATH), exist_ok=True)
    with open(LOG_EVENTS_PATH, "a") as f:
        f.write(json.dumps(evento) + "\n")
    logger.info(json.dumps(evento))


def _particion() -> str:
    """Fecha logica = INICIO del periodo semanal (donde se escribe el resultado)."""
    context = get_current_context()
    return context["logical_date"].in_timezone("America/Mexico_City").format("YYYY/MM/DD")


def _tope_datos() -> str:
    """FIN del periodo semanal: la corrida "del 11" se ejecuta el 18 y puede
    usar cualquier particion Silver hasta ese dia (la mas reciente)."""
    context = get_current_context()
    return context["data_interval_end"].in_timezone("America/Mexico_City").format("YYYY/MM/DD")


@dag(
    dag_id="entrenamiento_modelo_dag",
    description="S14-S15: dataset de features + entrenamiento, evaluacion y registro del modelo de riesgo",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 10, 1, tz="America/Mexico_City"),
    catchup=False,
    default_args=default_args,
    tags=["ml", "features", "modelo", "gemelo-digital-financiero"],
)
def entrenamiento_modelo_dag():

    @task
    def construir_dataset_features() -> dict:
        particion = _particion()
        try:
            metadata = construir_dataset_entrenamiento(particion, _tope_datos())
        except FileNotFoundError as error:
            raise AirflowSkipException(
                f"No hay ninguna particion de credit_risk en Silver hasta {particion}: corre "
                f"downloadRawData.py y la malla Bronze/Silver. Detalle: {error}"
            )
        registrar_evento({
            "nivel": "INFO",
            "task": "construir_dataset_features",
            "fuente": "gold_features_entrenamiento",
            "particion": particion,
            "version_dataset": metadata["version_dataset"],
            "particion_origen": metadata["particion_origen"],
            "filas": metadata["filas"],
            "features_seleccionadas": metadata["features_seleccionadas"],
        })
        return {"version_dataset": metadata["version_dataset"], "filas": metadata["filas"]}

    @task
    def entrenar_y_registrar_modelo(_dataset: dict) -> dict:
        particion = _particion()
        resumen = entrenar_y_registrar(particion, _tope_datos())
        registrar_evento({
            # Un modelo que no se promueve no es una falla de la malla, pero
            # si algo que revisar: se registra como ALERTA para que se vea.
            "nivel": "INFO" if resumen["promovido"] else "ALERTA",
            "task": "entrenar_y_registrar",
            "fuente": "ml-models/riesgo_credito",
            "particion": particion,
            **resumen,
        })
        return resumen

    entrenar_y_registrar_modelo(construir_dataset_features())


entrenamiento_modelo_dag()
