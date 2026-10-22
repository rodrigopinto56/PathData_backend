"""
dags/espera_dags.py

Robustez de la malla (post S15): cada DAG diario espera a que el anterior
(Bronze -> Silver -> Gold) haya terminado con exito EL MISMO DIA antes de
arrancar. Sin esto, los tres arrancan a la misma hora (medianoche de Mexico)
y Silver/Gold podian correr antes de que existieran sus datos de entrada.

No es un DAG: es un modulo auxiliar que importan transformacion_silver_dag y
transformacion_gold_dag (Airflow agrega la carpeta dags/ al sys.path).

Por que no un ExternalTaskSensor "de libro": ese sensor busca una corrida del
DAG anterior con EXACTAMENTE la misma fecha logica. Las corridas programadas
coinciden (todas a las 00:00), pero un "Trigger DAG" manual tiene como fecha
logica la hora del clic (p.ej. 10:37:12) y el sensor se quedaba esperando una
corrida que nunca iba a existir. Aqui basta con que el DAG anterior tenga
una corrida exitosa en el mismo DIA (hora de Mexico), sea programada,
backfill o manual.
"""

from __future__ import annotations

import logging

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import DagRun
from airflow.operators.python import get_current_context
from airflow.utils.state import DagRunState

logger = logging.getLogger("espera_dags")

ZONA = "America/Mexico_City"


def esperar_dag(task_id: str, dag_previo: str):
    """Sensor que espera una corrida EXITOSA de `dag_previo` el mismo dia.

    - mode="reschedule": mientras espera no ocupa un worker.
    - Si todas las corridas del DAG anterior ese dia FALLARON, falla de
      inmediato con un mensaje claro (en vez de esperar horas).
    - Escape para pruebas: disparar con config {"sin_esperar": true} salta la
      espera (util para re-procesar solo Gold con datos que ya existen).
    """

    @task.sensor(task_id=task_id, mode="reschedule", poke_interval=120, timeout=6 * 60 * 60)
    def _esperar() -> bool:
        context = get_current_context()
        conf = context["dag_run"].conf or {}
        if conf.get("sin_esperar"):
            logger.info("sin_esperar=true: no se espera a %s", dag_previo)
            return True

        inicio = context["logical_date"].in_timezone(ZONA).start_of("day")
        fin = inicio.end_of("day")
        corridas = DagRun.find(
            dag_id=dag_previo, execution_start_date=inicio, execution_end_date=fin
        )
        estados = [corrida.state for corrida in corridas]
        logger.info("Corridas de %s el %s: %s", dag_previo, inicio.to_date_string(), estados)

        if DagRunState.SUCCESS in estados:
            return True
        if estados and all(estado == DagRunState.FAILED for estado in estados):
            raise AirflowFailException(
                f"{dag_previo} fallo el {inicio.to_date_string()}: corrigelo, vuelve a "
                f"correrlo y luego limpia (Clear) esta tarea para reintentar."
            )
        return False  # todavia no corre o sigue corriendo: volver a revisar

    return _esperar()
