"""
dashboard/data_loader.py

Su  trabajo es leer las dos fuentes
que ya genera tu malla de datos y devolverlas como DataFrames de pandas,
listos para que app.py los grafique:

  1) data/reports/pipeline_log.jsonl
     -> Log de eventos JSON Lines que ya escriben ingesta_bronze_dag.py
        y transformacion_silver_dag.py. Aqui vive la "salud" de la malla:
        exito/fallo de tareas, fechas, fuentes

  2) silver-layer (MinIO), archivos _calidad_reporte.json y
     _silver_metadata.json
     -> Uno por cada particion (fuente + fecha). Aqui vive la "calidad"
        de los datos: cuantas expectativas de Great Expectations pasaron
"""

from __future__ import annotations

import json
import os

import pandas as pd
import s3fs

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "password123")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver-layer")

PIPELINE_LOG_PATH = "/app/data/reports/pipeline_log.jsonl"


def _fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        key=MINIO_KEY,
        secret=MINIO_SECRET,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )


def cargar_eventos_pipeline(ruta: str = PIPELINE_LOG_PATH) -> pd.DataFrame:
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
                continue

    df = pd.DataFrame(eventos)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _listar_archivos(sufijo: str) -> list[str]:
    fs = _fs()
    prefijo = f"{SILVER_BUCKET}/"
    if not fs.exists(prefijo):
        return []
    todas = fs.find(prefijo)
    return [ruta for ruta in todas if ruta.endswith(sufijo)]


def cargar_reportes_calidad() -> pd.DataFrame:
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