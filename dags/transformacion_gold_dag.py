"""
dags/transformacion_gold_dag.py

Semanas 11-12: orquesta Silver -> Gold.
  1) Entrena el modelo de riesgo sobre la particion del dia de
     loan_default_risk.
  2) Califica (infiere riesgo) a los usuarios de personal_finance_ml.
  3) Construye el modelo estrella (dim_* + fact_posicion_financiera),
     valida su integridad y lo sube a MinIO junto con
     gold_clientes_riesgo + gold_metricas_por_segmento.
  4) (Semana 12) Calcula los KPIs financieros -- riesgo, exposicion y
     comportamiento transaccional -- sobre el modelo estrella y los sube
     como gold_kpis_financieros + gold_kpis_por_segmento.

A diferencia de transformacion_silver_dag, este DAG NO usa
SparkSubmitOperator: toda la logica corre en pandas dentro de la misma
tarea de Airflow. Ver scripts/construir_gold.py para el por que (el
volumen de datos no justifica un cluster de Spark).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pendulum

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

sys.path.append("/opt/airflow/scripts")
from construir_gold import construir_gold_particion
from kpis_financieros import calcular_kpis_particion

logger = logging.getLogger("transformacion_gold_dag")

LOG_EVENTS_PATH = "/opt/airflow/data/reports/pipeline_log.jsonl"

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
}


def registrar_evento(evento: dict) -> None:
    """Mismo log compartido que ya usan ingesta_bronze_dag y
    transformacion_silver_dag -- una sola fuente de eventos para todo
    el dashboard de calidad."""
    evento["timestamp"] = pendulum.now("America/Mexico_City").isoformat()
    os.makedirs(os.path.dirname(LOG_EVENTS_PATH), exist_ok=True)
    with open(LOG_EVENTS_PATH, "a") as f:
        f.write(json.dumps(evento) + "\n")
    logger.info(json.dumps(evento))


@dag(
    dag_id="transformacion_gold_dag",
    description="Silver -> Gold: modelo de riesgo, modelo estrella y KPIs financieros",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 9, 1, tz="America/Mexico_City"),
    catchup=False,
    default_args=default_args,
    tags=["gold", "ml", "kpis", "gemelo-digital-financiero"],
)
def transformacion_gold_dag():

    @task
    def construir_gold() -> dict:
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        resumen = construir_gold_particion(particion)

        registrar_evento(
            {
                "nivel": "INFO",
                "task": "construir_gold",
                "fuente": "modelo_estrella+gold_clientes_riesgo+gold_metricas_por_segmento",
                "particion": particion,
                "accuracy_modelo": resumen["modelo"]["accuracy"],
                "integridad_modelo_estrella": resumen["integridad_modelo_estrella"]["exito_global"],
            }
        )

        return resumen

    @task
    def calcular_kpis() -> dict:
        """Semana 12. Corre despues de construir_gold porque lee el modelo
        estrella que esa tarea acaba de escribir.

        El evento se registra como INFO aunque haya KPIs fuera de umbral:
        una alerta de negocio (p.ej. mucha exposicion en riesgo alto) no
        es una falla de la malla, y no debe bajar la "tasa de exito" de la
        Vista 1. Los KPIs en alerta se ven en la Vista 4 del dashboard."""
        context = get_current_context()
        fecha_logica = context["logical_date"].in_timezone("America/Mexico_City")
        particion = fecha_logica.format("YYYY/MM/DD")

        resumen = calcular_kpis_particion(particion)

        registrar_evento(
            {
                "nivel": "INFO",
                "task": "calcular_kpis",
                "fuente": "gold_kpis_financieros+gold_kpis_por_segmento",
                "particion": particion,
                "kpis_calculados": resumen["kpis_calculados"],
                "kpis_en_alerta": resumen["kpis_en_alerta"],
            }
        )

        return resumen

    construir_gold() >> calcular_kpis()


transformacion_gold_dag()