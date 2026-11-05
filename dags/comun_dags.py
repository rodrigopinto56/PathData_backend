"""
dags/comun_dags.py

Semana 18 (hardening): configuracion comun de robustez para todos los DAGs.
No es un DAG; lo importan los DAGs (Airflow agrega dags/ al sys.path).

  - execution_timeout: ninguna tarea se queda colgada para siempre; si se
    pasa del limite, falla (y reintenta) en vez de bloquear la malla.
  - on_failure_callback: cuando una tarea falla DEFINITIVAMENTE (agoto sus
    reintentos) se registra un evento "falla_tarea" en pipeline_log.jsonl,
    que la Vista 0 del dashboard tecnico muestra como alerta.
"""

from __future__ import annotations

import json
import logging
import os

import pendulum

LOG_EVENTS_PATH = "/opt/airflow/data/reports/pipeline_log.jsonl"
logger = logging.getLogger("comun_dags")


def alertar_fallo(context: dict) -> None:
    ti = context["task_instance"]
    evento = {
        "nivel": "ALERTA",
        "tipo": "falla_tarea",
        "dag_id": ti.dag_id,
        "task_id": ti.task_id,
        "run_id": ti.run_id,
        "logical_date": str(context.get("logical_date")),
        "intento": ti.try_number,
        "error": str(context.get("exception"))[:500],
        "log_url": ti.log_url,
        "timestamp": pendulum.now("America/Mexico_City").isoformat(),
    }
    try:
        os.makedirs(os.path.dirname(LOG_EVENTS_PATH), exist_ok=True)
        with open(LOG_EVENTS_PATH, "a") as f:
            f.write(json.dumps(evento) + "\n")
    except OSError:
        pass  # el aviso nunca debe tapar el error original
    logger.error(json.dumps(evento))


def argumentos_base(reintentos: int = 2, timeout_min: int = 60) -> dict:
    """default_args comunes: reintentos, limite de tiempo y aviso de falla."""
    return {
        "owner": "data-engineering",
        "retries": reintentos,
        "retry_delay": pendulum.duration(minutes=5),
        "execution_timeout": pendulum.duration(minutes=timeout_min),
        "on_failure_callback": alertar_fallo,
    }
