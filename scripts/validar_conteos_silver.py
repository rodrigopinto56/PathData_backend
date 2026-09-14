"""
scripts/validar_conteos_silver.py

Valida que los conteos de filas en Silver (valid vs quarantine) tengan
sentido de negocio, leyendo directamente desde MinIO (no desde disco local,
ya que el pipeline sube todo ahi via la tarea 'subir_silver_a_minio').

Uso:
    docker compose exec airflow-scheduler python /opt/airflow/data/scripts/validar_conteos_silver.py
"""

from __future__ import annotations

import os
import pandas as pd

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "password123")
BUCKET = os.getenv("BRONZE_BUCKET_SILVER", "silver-layer")  

STORAGE_OPTIONS = {
    "key": MINIO_KEY,
    "secret": MINIO_SECRET,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}

FUENTES = ["loan_default_risk", "personal_finance_ml"]
FECHAS = ["2026/09/01", "2026/09/02"]
CAPAS = ["valid", "quarantine"]


def contar_filas(fuente: str, fecha: str, capa: str) -> int | None:
    """Lee todos los parquet de una particion (carpeta) y regresa el total de filas.
    Regresa None si la ruta no existe o esta vacia (no lo trata como error fatal).
    """
    ruta = f"s3://{BUCKET}/{fuente}/{fecha}/{capa}"
    try:
        df = pd.read_parquet(ruta, storage_options=STORAGE_OPTIONS)
        return len(df)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"  [ERROR leyendo {ruta}]: {e}")
        return None


def main() -> None:
    print(f"Conectando a MinIO en {MINIO_ENDPOINT}, bucket '{BUCKET}'\n")

    resumen = []
    for fuente in FUENTES:
        for fecha in FECHAS:
            fila = {"fuente": fuente, "fecha": fecha}
            for capa in CAPAS:
                fila[capa] = contar_filas(fuente, fecha, capa)
            resumen.append(fila)

    print(f"{'fuente':<22} {'fecha':<12} {'valid':>10} {'quarantine':>12} {'%cuarentena':>13}")
    print("-" * 72)
    for fila in resumen:
        valid = fila["valid"] or 0
        quarantine = fila["quarantine"] or 0
        total = valid + quarantine
        pct = (quarantine / total * 100) if total > 0 else 0.0
        print(f"{fila['fuente']:<22} {fila['fecha']:<12} {valid:>10} {quarantine:>12} {pct:>12.2f}%")

    print("\nReferencia esperada segun dataCatalog.yml:")
    print("  - personal_finance_ml: quarantine debe ser MUY bajo (el 60% de loan_type nulo NO cuenta)")
    print("  - loan_default_risk:   quarantine ~0.33%-0.67% (nulos reales en Debt_Amount/Monthly_Savings)")


if __name__ == "__main__":
    main()