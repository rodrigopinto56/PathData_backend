# pip install kagglehub --break-system-packages

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import kagglehub


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Solo los 2 datasets del MVP 
DATASETS = {
    "loan_default_risk_dataset.csv": {
        "kaggle_id": "himelsarder/loan-default-risk-prediction-dataset",
        "archivo_origen": "loan_default_risk_dataset.csv",
    },
    "synthetic_personal_finance_dataset.csv": {
        "kaggle_id": "miadul/personal-finance-ml-dataset",
        "archivo_origen": "synthetic_personal_finance_dataset.csv",
    },
    # Semana 14: fuentes para IA. Son "opcionales": si la descarga falla,
    # se avisa y se sigue con las demas (el pipeline las salta si no estan).
    # Riesgo crediticio etiquetado (~32K prestamos con loan_status 0/1):
    # base para entrenar el modelo predictivo de la Semana 15.
    "credit_risk_dataset.csv": {
        "kaggle_id": "laotse/credit-risk-dataset",
        "archivo_origen": "credit_risk_dataset.csv",
        "opcional": True,
    },
    # Transacciones de un cliente (60 meses): serie de tiempo para la
    # prediccion financiera y el KPI de estabilidad (cliente demo).
    "personal_finance_transactions.csv": {
        "kaggle_id": "ramyapintchy/personal-finance-data",
        "archivo_origen": "Personal_Finance_Dataset.csv",
        "opcional": True,
    },
}


def sha256_de(path: Path) -> str:
    """Checksum del archivo -- permite detectar despues si Kaggle actualizo
    el dataset y tu copia local quedo desincronizada."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            h.update(bloque)
    return h.hexdigest()


def main() -> None:
    manifiesto = {}

    for nombre_local, cfg in DATASETS.items():
        print(f"Descargando {cfg['kaggle_id']} ...")
        try:
            ruta_cache = Path(kagglehub.dataset_download(cfg["kaggle_id"]))
            origen = ruta_cache / cfg["archivo_origen"]
            if not origen.exists():
                encontrados = [p.name for p in ruta_cache.glob("*.csv")]
                raise FileNotFoundError(
                    f"No se encontro '{cfg['archivo_origen']}' en {ruta_cache} "
                    f"(CSVs encontrados: {encontrados}). Revisa el nombre exacto del archivo."
                )
        except Exception as error:  # noqa: BLE001 - se re-lanza si la fuente es obligatoria
            if not cfg.get("opcional"):
                raise
            print(f"  AVISO: se omite la fuente opcional {nombre_local}: {error}")
            continue

        destino = RAW_DIR / nombre_local
        shutil.copyfile(origen, destino)

        with open(destino) as f:
            total_filas = sum(1 for _ in f) - 1  # resta encabezado

        manifiesto[nombre_local] = {
            "kaggle_id": cfg["kaggle_id"],
            "sha256": sha256_de(destino),
            "descargado_en": datetime.now(timezone.utc).isoformat(),
            "filas": total_filas,
        }
        print(f"  -> {destino}  ({total_filas} filas)")

    with open(RAW_DIR / "manifest.json", "w") as f:
        json.dump(manifiesto, f, indent=2)

    print(f"\nManifiesto guardado en {RAW_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
