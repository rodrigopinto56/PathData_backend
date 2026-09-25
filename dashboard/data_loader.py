"""
dashboard/data_loader.py

Paso 3 del plan (Semana 10): capa de datos del dashboard.

Este archivo NO dibuja nada. Su unico trabajo es leer las dos fuentes
que ya genera tu malla de datos y devolverlas como DataFrames de pandas,
listos para que app.py los grafique:

  1) data/reports/pipeline_log.jsonl
     -> Log de eventos JSON Lines que ya escriben ingesta_bronze_dag.py
        y transformacion_silver_dag.py. Aqui vive la "salud" de la malla:
        exito/fallo de tareas, fechas, fuentes.

  2) silver-layer (MinIO), archivos _calidad_reporte.json y
     _silver_metadata.json
     -> Uno por cada particion (fuente + fecha). Aqui vive la "calidad"
        de los datos: cuantas expectativas de Great Expectations pasaron.

Separar esto de app.py es a proposito: si mañana cambias de Streamlit a
otra cosa, o quieres testear la logica de carga con pytest, no tienes
que tocar ni una linea de interfaz.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import s3fs

# --- Configuracion (mismas variables que ya usa validar_calidad_silver.py) ---
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "password123")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver-layer")
GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold-layer")

# Ruta local del log de eventos. Se monta desde el host con el mismo
# volumen ./data que ya usan tus otros contenedores (ver docker-compose.yml,
# servicio streamlit -> volumes: ./data:/app/data).
PIPELINE_LOG_PATH = "/app/data/reports/pipeline_log.jsonl"


def _fs() -> s3fs.S3FileSystem:
    """Crea una conexion a MinIO reutilizando las credenciales del stack.

    Se crea una nueva conexion cada vez que se llama (en vez de una
    global) porque s3fs cachea listados internamente; para un dashboard
    que se refresca seguido, una conexion nueva evita que veamos datos
    viejos si el usuario aprieta 'rerun' en Streamlit.
    """
    return s3fs.S3FileSystem(
        key=MINIO_KEY,
        secret=MINIO_SECRET,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )


def cargar_eventos_pipeline(ruta: str = PIPELINE_LOG_PATH) -> pd.DataFrame:
    """Lee pipeline_log.jsonl y regresa un DataFrame, uno renglon por evento.

    Si el archivo todavia no existe (por ejemplo, primera vez que se
    levanta el stack y nadie ha corrido un DAG), regresamos un
    DataFrame vacio en vez de tronar: el dashboard debe poder abrirse
    aunque todavia no haya datos que mostrar.
    """
    if not os.path.exists(ruta):
        return pd.DataFrame(
            columns=["timestamp", "nivel", "task", "fuente", "particion"]
        )

    eventos = []
    with open(ruta) as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            try:
                eventos.append(json.loads(linea))
            except json.JSONDecodeError:
                # Una linea corrupta (por ejemplo, un corte de luz a
                # medio-escribir) no debe tumbar todo el dashboard.
                continue

    df = pd.DataFrame(eventos)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _listar_archivos(sufijo: str, bucket: str = SILVER_BUCKET) -> list[str]:
    """Regresa las rutas s3:// de todos los archivos en `bucket`
    que terminan en `sufijo` (por ejemplo '_calidad_reporte.json').

    Recibe el bucket como parametro (antes solo funcionaba con
    silver-layer) porque la Semana 11 agrego un segundo bucket,
    gold-layer, con la misma necesidad de "listar y filtrar por sufijo"."""
    fs = _fs()
    prefijo = f"{bucket}/"
    if not fs.exists(prefijo):
        return []
    todas = fs.find(prefijo)
    return [ruta for ruta in todas if ruta.endswith(sufijo)]


def cargar_reportes_calidad() -> pd.DataFrame:
    """Recorre silver-layer, lee cada _calidad_reporte.json y regresa
    un DataFrame con una fila por particion (fuente + fecha)."""
    fs = _fs()
    rutas = _listar_archivos("_calidad_reporte.json")

    filas = []
    for ruta in rutas:
        with fs.open(ruta, "r") as f:
            reporte = json.load(f)
        filas.append(
            {
                "fuente": reporte.get("fuente"),
                "fecha": reporte.get("fecha"),
                "exito_global": reporte.get("exito_global"),
                "expectativas_totales": reporte.get("expectativas_totales"),
                "expectativas_exitosas": reporte.get("expectativas_exitosas"),
                "expectativas_fallidas": len(reporte.get("expectativas_fallidas", [])),
                "filas_evaluadas": reporte.get("filas_evaluadas"),
            }
        )

    df = pd.DataFrame(filas)
    if not df.empty:
        df["porcentaje_calidad"] = (
            df["expectativas_exitosas"] / df["expectativas_totales"] * 100
        ).round(1)
    return df


def cargar_metadata_silver() -> pd.DataFrame:
    """Recorre silver-layer, lee cada _silver_metadata.json (el 'congelado'
    de la Semana 11) y regresa un DataFrame con una fila por particion."""
    fs = _fs()
    rutas = _listar_archivos("_silver_metadata.json")

    filas = []
    for ruta in rutas:
        with fs.open(ruta, "r") as f:
            metadata = json.load(f)
        calidad = metadata.get("calidad", {})
        filas.append(
            {
                "fuente": metadata.get("fuente"),
                "particion": metadata.get("particion"),
                "version_dataset": metadata.get("version_dataset"),
                "congelado_en": metadata.get("congelado_en"),
                "exito_global": calidad.get("exito_global"),
                "filas_evaluadas": calidad.get("filas_evaluadas"),
            }
        )

    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Capa Gold (Semana 11) -- ver config/gold_schema.yml para el diseno
# ---------------------------------------------------------------------------
def cargar_gold_metadata_modelo() -> pd.DataFrame:
    """Recorre gold-layer, lee cada _gold_metadata/<fecha>.json y regresa
    una fila por particion con las metricas del modelo de riesgo
    (accuracy, AUC) que ya vimos al validar el backfill."""
    fs = _fs()
    rutas = _listar_archivos(".json", bucket=GOLD_BUCKET)
    rutas = [r for r in rutas if "_gold_metadata" in r]

    filas = []
    for ruta in rutas:
        with fs.open(ruta, "r") as f:
            metadata = json.load(f)
        modelo = metadata.get("modelo", {})
        filas.append(
            {
                "fecha": metadata.get("fecha"),
                "accuracy": modelo.get("accuracy"),
                "auc": modelo.get("auc"),
                "filas_entrenamiento": modelo.get("filas_entrenamiento"),
                "filas_prueba": modelo.get("filas_prueba"),
                "clientes_calificados": metadata.get("gold_clientes_riesgo", {}).get("filas"),
            }
        )

    return pd.DataFrame(filas)


def cargar_gold_metricas_segmento() -> pd.DataFrame:
    """Recorre gold-layer, lee cada particion de gold_metricas_por_segmento
    y las junta en un solo DataFrame (una fila por region x banda x fecha)."""
    fs = _fs()
    prefijo = f"{GOLD_BUCKET}/gold_metricas_por_segmento/"
    if not fs.exists(prefijo):
        return pd.DataFrame()

    rutas_parquet = [r for r in fs.find(prefijo) if r.endswith(".parquet")]
    if not rutas_parquet:
        return pd.DataFrame()

    partes = [pd.read_parquet(f"s3://{r}", storage_options=_storage_options()) for r in rutas_parquet]
    return pd.concat(partes, ignore_index=True)


def cargar_gold_clientes_riesgo() -> pd.DataFrame:
    """Recorre gold-layer, lee cada particion de gold_clientes_riesgo (el
    detalle por cliente) y las junta en un solo DataFrame."""
    fs = _fs()
    prefijo = f"{GOLD_BUCKET}/gold_clientes_riesgo/"
    if not fs.exists(prefijo):
        return pd.DataFrame()

    rutas_parquet = [r for r in fs.find(prefijo) if r.endswith(".parquet")]
    if not rutas_parquet:
        return pd.DataFrame()

    partes = [pd.read_parquet(f"s3://{r}", storage_options=_storage_options()) for r in rutas_parquet]
    return pd.concat(partes, ignore_index=True)


def _storage_options() -> dict:
    return {
        "key": MINIO_KEY,
        "secret": MINIO_SECRET,
        "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
    }