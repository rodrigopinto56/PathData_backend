"""
Chequeo ligero de salud de la capa Bronze. Para cada fuente:
  1) Recorre las particiones year/month/day existentes.
  2) Valida que cada particion tenga sus 2 archivos esperados
     (data.parquet + _ingestion_metadata.json), y que el JSON no este corrupto.
  3) Detecta HUECOS: fechas faltantes dentro del rango min-max encontrado
     (util para confirmar que un backfill quedo completo, o para detectar
     dias en los que el DAG fallo silenciosamente).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

BRONZE_DIR = Path("data/bronze")


def particiones_de(fuente_dir: Path) -> list[date]:
    fechas = []
    for year_dir in sorted(fuente_dir.glob("*")):
        for month_dir in sorted(year_dir.glob("*")):
            for day_dir in sorted(month_dir.glob("*")):
                try:
                    fechas.append(date(int(year_dir.name), int(month_dir.name), int(day_dir.name)))
                except ValueError:
                    continue
    return fechas


def validar_particion(dia_dir: Path) -> list[str]:
    problemas = []
    if not (dia_dir / "data.parquet").exists():
        problemas.append("falta data.parquet")

    meta_path = dia_dir / "_ingestion_metadata.json"
    if not meta_path.exists():
        problemas.append("falta _ingestion_metadata.json")
    else:
        try:
            with open(meta_path) as f:
                json.load(f)
        except json.JSONDecodeError:
            problemas.append("_ingestion_metadata.json corrupto")

    return problemas


def main() -> None:
    if not BRONZE_DIR.exists():
        print(f"No existe {BRONZE_DIR} -- corre el DAG primero.")
        return

    for fuente_dir in sorted(BRONZE_DIR.iterdir()):
        if not fuente_dir.is_dir():
            continue

        print(f"\n=== {fuente_dir.name} ===")
        fechas = particiones_de(fuente_dir)

        if not fechas:
            print("  Sin particiones encontradas.")
            continue

        total_problemas = 0
        for f in fechas:
            dia_dir = fuente_dir / f"{f.year:04d}" / f"{f.month:02d}" / f"{f.day:02d}"
            problemas = validar_particion(dia_dir)
            if problemas:
                total_problemas += 1
                print(f"  [{f}] PROBLEMA: {', '.join(problemas)}")

        fecha_min, fecha_max = min(fechas), max(fechas)
        rango_completo = {
            fecha_min + timedelta(days=i)
            for i in range((fecha_max - fecha_min).days + 1)
        }
        huecos = sorted(rango_completo - set(fechas))

        print(f"  Rango: {fecha_min} a {fecha_max}  ({len(fechas)} particiones encontradas)")
        if huecos:
            print(f"  HUECOS detectados ({len(huecos)}): {[str(h) for h in huecos]}")
        else:
            print("  Sin huecos en el rango.")

        if total_problemas == 0 and not huecos:
            print("  Estado: OK")


if __name__ == "__main__":
    main()