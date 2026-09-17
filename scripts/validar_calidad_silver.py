"""
scripts/validar_calidad_silver.py

Semana 11: Framework de calidad declarativa (Great Expectations).

Lee config/silver_schema.yml (el contrato de esquema definido en la
Semana 10) y genera automaticamente un Expectation Suite por fuente,
evitando declarar las mismas reglas dos veces (una en PySpark, otra en GE).

Expone:
  - validar_particion(fuente, fecha) -> dict
        Funcion reutilizable, pensada para ser importada y llamada desde
        una tarea de Airflow (Semana 11, paso 5) para UNA sola particion.
  - main()
        Modo CLI manual: sin argumentos, barre todas las fuentes y fechas
        de prueba (comportamiento original, util para validaciones ad-hoc
        fuera de Airflow).

Uso manual (sin Airflow):
    python validar_calidad_silver.py

Uso como libreria (dentro de un DAG):
    import sys; sys.path.append("/opt/airflow/scripts")
    from validar_calidad_silver import validar_particion
    resultado = validar_particion("loan_default_risk", "2026/09/01")
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import yaml
import great_expectations as ge

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "password123")
BUCKET = os.getenv("SILVER_BUCKET", "silver-layer")

STORAGE_OPTIONS = {
    "key": MINIO_KEY,
    "secret": MINIO_SECRET,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}

SCHEMA_PATH = "/opt/airflow/config/silver_schema.yml"
FECHAS_PRUEBA = ["2026/09/01", "2026/09/02"]  
REPORTE_SALIDA_CLI = "/opt/airflow/data/reporte_calidad_silver.json"


def cargar_contrato() -> dict:
    with open(SCHEMA_PATH) as f:
        return yaml.safe_load(f)["fuentes"]


def construir_expectativas(df_ge, columnas: list[dict]) -> None:
    """Traduce cada columna del contrato en expectativas de Great Expectations."""
    for col in columnas:
        nombre = col["nombre"]
        if nombre not in df_ge.columns:
            continue  

        if not col.get("nulable", True):
            df_ge.expect_column_values_to_not_be_null(nombre)

        regla = col.get("regla", "")

        if ">= 0" in regla:
            df_ge.expect_column_values_to_be_between(nombre, min_value=0, mostly=1.0)
        elif "> 0" in regla:
            df_ge.expect_column_values_to_be_between(
                nombre, min_value=0, strict_min=True, mostly=1.0
            )
        elif "entre 300 y 850" in regla or "300 y 850" in regla:
            df_ge.expect_column_values_to_be_between(nombre, min_value=300, max_value=850)
        elif "en {0, 1}" in regla:
            df_ge.expect_column_values_to_be_in_set(nombre, [0, 1])
        elif "en {Si, No}" in regla or "en {Yes, No}" in regla:
            df_ge.expect_column_values_to_be_in_set(nombre, ["Si", "No", "Yes"])


def validar_particion(fuente: str, fecha: str) -> dict:
    """Corre el Expectation Suite de una fuente contra UNA sola particion
    (fecha) de la capa Silver (valid/) en MinIO. Pensada para llamarse
    desde una tarea de Airflow, una vez por corrida del DAG.
    """
    contrato = cargar_contrato()
    contrato_fuente = contrato[fuente]

    ruta = f"s3://{BUCKET}/{fuente}/{fecha}/valid"
    df = pd.read_parquet(ruta, storage_options=STORAGE_OPTIONS)

    df_ge = ge.from_pandas(df)
    construir_expectativas(df_ge, contrato_fuente["columnas"])

    llave = contrato_fuente.get("llave_deduplicacion")
    if llave and llave in df.columns:
        df_ge.expect_column_values_to_be_unique(llave)

    resultado = df_ge.validate()

    return {
        "fuente": fuente,
        "fecha": fecha,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "filas_evaluadas": len(df),
        "exito_global": resultado["success"],
        "expectativas_totales": len(resultado["results"]),
        "expectativas_exitosas": sum(1 for r in resultado["results"] if r["success"]),
        "expectativas_fallidas": [
            {
                "expectativa": r["expectation_config"]["expectation_type"],
                "columna": r["expectation_config"]["kwargs"].get("column"),
                "detalle": r["result"],
            }
            for r in resultado["results"] if not r["success"]
        ],
    }


def main() -> None:
    """Modo CLI manual: barre todas las fuentes y fechas de prueba."""
    contrato = cargar_contrato()
    reporte = {"generado_en": datetime.now(timezone.utc).isoformat(), "fuentes": []}

    for nombre_fuente in contrato:
        print(f"Validando {nombre_fuente} ...")
        resultados_por_fecha = []

        for fecha in FECHAS_PRUEBA:
            try:
                resultado = validar_particion(nombre_fuente, fecha)
            except FileNotFoundError:
                continue

            resultados_por_fecha.append(resultado)
            estado = "OK" if resultado["exito_global"] else "FALLAS DETECTADAS"
            print(
                f"  [{fecha}] {estado} - "
                f"{resultado['expectativas_exitosas']}/{resultado['expectativas_totales']} "
                f"expectativas cumplidas ({resultado['filas_evaluadas']} filas)"
            )
            for falla in resultado["expectativas_fallidas"]:
                print(f"    FALLO: {falla['expectativa']} en columna '{falla['columna']}'")

        reporte["fuentes"].append({"fuente": nombre_fuente, "particiones": resultados_por_fecha})

    os.makedirs(os.path.dirname(REPORTE_SALIDA_CLI), exist_ok=True)
    with open(REPORTE_SALIDA_CLI, "w") as f:
        json.dump(reporte, f, indent=2, default=str)

    print(f"\nReporte guardado en {REPORTE_SALIDA_CLI}")


if __name__ == "__main__":
    main()