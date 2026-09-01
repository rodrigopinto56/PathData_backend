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
        ruta_cache = Path(kagglehub.dataset_download(cfg["kaggle_id"]))
        origen = ruta_cache / cfg["archivo_origen"]

        if not origen.exists():
            raise FileNotFoundError(
                f"No se encontro '{cfg['archivo_origen']}' en {ruta_cache}. "
                f"Revisa el nombre exacto del archivo dentro del dataset de Kaggle."
            )

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
