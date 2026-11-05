"""
dags/monitoreo_dag.py

Semana 18: observabilidad. Cada hora:

  1) exportar_metricas: lee de la base de metadatos de Airflow las corridas
     (DagRun) y tareas (TaskInstance) de los ultimos 30 dias y las deja en
     data/reports/observabilidad/*.parquet. El dashboard tecnico no tiene
     acceso a la base de Airflow; con esto, si.
  2) evaluar_slas: evalua los SLA de config/slas.yml (mismo codigo que el
     dashboard: scripts/observabilidad.py) y registra un evento en
     pipeline_log.jsonl SOLO cuando un SLA cambia de estado (OK -> ALERTA o
     de vuelta), para no llenar el log con la misma alerta cada hora.

Se excluye a si mismo de las metricas (sus corridas horarias no son parte
de la malla de datos).
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd
import pendulum

from airflow.decorators import dag, task

sys.path.append("/opt/airflow/scripts")
from comun_dags import argumentos_base  # dags/comun_dags.py
import observabilidad as obs

logger = logging.getLogger("monitoreo_dag")
DAG_ID = "monitoreo_dag"


def _corridas_y_tareas(dias: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from airflow.models import DagRun, TaskInstance
    from airflow.utils.session import create_session

    desde = pendulum.now("UTC").subtract(days=dias)
    with create_session() as session:
        corridas = session.query(DagRun).filter(DagRun.dag_id != DAG_ID, DagRun.start_date >= desde).all()
        tareas = (session.query(TaskInstance)
                  .filter(TaskInstance.dag_id != DAG_ID, TaskInstance.start_date >= desde).all())
        df_corridas = pd.DataFrame([{
            "dag_id": c.dag_id, "run_id": c.run_id, "run_type": c.run_type, "state": c.state,
            "logical_date": c.execution_date, "start_date": c.start_date, "end_date": c.end_date,
        } for c in corridas])
        df_tareas = pd.DataFrame([{
            "dag_id": t.dag_id, "task_id": t.task_id, "run_id": t.run_id, "map_index": t.map_index,
            "state": t.state, "start_date": t.start_date, "end_date": t.end_date,
            "duracion_seg": t.duration, "intentos": t.try_number, "operador": t.operator,
        } for t in tareas])
    return df_corridas, df_tareas


def _registrar_evento(evento: dict) -> None:
    evento["timestamp"] = pendulum.now("America/Mexico_City").isoformat()
    ruta = os.path.join(obs.CARPETA_REPORTES, "pipeline_log.jsonl")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "a") as f:
        f.write(json.dumps(evento, default=str) + "\n")
    logger.info(json.dumps(evento, default=str))


@dag(
    dag_id=DAG_ID,
    description="S18: exporta metricas operativas de Airflow y evalua los SLA de la plataforma",
    schedule="@hourly",
    start_date=pendulum.datetime(2026, 11, 1, tz="America/Mexico_City"),
    catchup=False,
    max_active_runs=1,
    default_args=argumentos_base(reintentos=1, timeout_min=10),
    tags=["observabilidad", "monitoreo", "gemelo-digital-financiero"],
)
def monitoreo_dag():

    @task
    def exportar_metricas() -> dict:
        config = obs.cargar_config()
        corridas, tareas = _corridas_y_tareas(config["historial_dias"])
        carpeta = obs.carpeta()
        os.makedirs(carpeta, exist_ok=True)
        corridas.to_parquet(os.path.join(carpeta, "corridas.parquet"), index=False)
        tareas.to_parquet(os.path.join(carpeta, "tareas.parquet"), index=False)
        resumen = {"generado_en": pendulum.now("America/Mexico_City").isoformat(),
                   "corridas": int(len(corridas)), "tareas": int(len(tareas))}
        with open(os.path.join(carpeta, "exportacion.json"), "w") as f:
            json.dump(resumen, f, indent=2)
        return resumen

    @task
    def evaluar_slas(_exportacion: dict) -> dict:
        import s3fs

        from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET

        config = obs.cargar_config()
        zona = config["zona_horaria"]
        ahora = pd.Timestamp.now(tz=zona)
        entradas = obs.cargar_entradas_locales()
        entradas["corridas"] = obs.preparar_corridas(entradas["corridas"], zona)
        fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT},
                               skip_instance_cache=True, use_listings_cache=False)
        entradas["calidad"] = obs.cargar_calidad_minio(fs, os.getenv("SILVER_BUCKET", "silver-layer"))
        entradas["modelo"] = obs.cargar_modelo_minio(fs, os.getenv("ML_BUCKET", "ml-models"))

        slas = obs.evaluar_slas(entradas, config, ahora)
        ruta_estado = os.path.join(obs.carpeta(), "slas_estado.json")
        anterior = json.load(open(ruta_estado)) if os.path.exists(ruta_estado) else {}
        for _, s in slas.iterrows():
            if anterior.get(s["sla_id"]) not in (None, s["estado"]) or (s["sla_id"] not in anterior and s["estado"] == obs.ALERTA):
                _registrar_evento({
                    "nivel": "ALERTA" if s["estado"] == obs.ALERTA else "INFO",
                    "tipo": "sla", "task": "evaluar_slas", "sla_id": s["sla_id"], "sla": s["nombre"],
                    "estado_anterior": anterior.get(s["sla_id"]), "estado": s["estado"],
                    "valor": s["valor"], "meta": s["meta"],
                })
        with open(ruta_estado, "w") as f:
            json.dump(dict(zip(slas["sla_id"], slas["estado"])), f, indent=2)
        return {"estados": dict(zip(slas["sla_id"], slas["estado"]))}

    evaluar_slas(exportar_metricas())


monitoreo_dag()
