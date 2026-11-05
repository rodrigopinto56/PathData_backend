"""
bronzeAudit.py (refactor - Semana 8; versionado en el repo desde la Semana 18)

Chequeo de salud de la capa Bronze en MinIO. Se mantiene como modulo
"puro": solo revisa y REGRESA datos estructurados -- quien decide que
hacer con el resultado (loguear, fallar la tarea, alertar) es el DAG,
no este modulo. Ademas usa la misma Connection de Airflow que
cargar_a_bronze -- nunca credenciales sueltas en el codigo.

Semana 18 (hardening): antes vivia solo en plugins/ (ignorado por Git), asi
que un clon nuevo del repo no podia cargar ingesta_bronze_dag. Ademas, el
listado ahora es paginado: list_objects_v2 regresa como maximo 1,000
objetos por llamada y, sin paginar, la auditoria ignoraba el resto.
"""

import os
from collections import defaultdict

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "bronze-layer")
S3_CONN_ID = os.getenv("BRONZE_S3_CONN_ID", "minio_s3_conn")
EXPECTED_FILES = {"data.parquet", "_ingestion_metadata.json"}


def agrupar_particiones(llaves: list[str]) -> dict[str, set]:
    """'fuente/AAAA/MM/DD/archivo' -> {'fuente/AAAA/MM/DD': {archivos}}"""
    partitions = defaultdict(set)
    for key in llaves:
        parts = key.rsplit("/", 1)
        if len(parts) == 2:
            partition_path, file_name = parts
            partitions[partition_path].add(file_name)
    return partitions


def revisar_particiones(partitions: dict[str, set]) -> list[dict]:
    problemas = []
    for partition, files in sorted(partitions.items()):
        if files != EXPECTED_FILES:
            missing = EXPECTED_FILES - files
            extra = files - EXPECTED_FILES
            detalle = []
            if missing:
                detalle.append(f"faltan {missing}")
            if extra:
                detalle.append(f"sobran {extra}")
            problemas.append({"particion": partition, "detalle": "; ".join(detalle)})
    return problemas


def audit_bronze_layer() -> dict:
    print(f"Iniciando auditoria del bucket: '{BRONZE_BUCKET}'...\n")

    hook = S3Hook(aws_conn_id=S3_CONN_ID)
    client = hook.get_conn()

    llaves = []
    for pagina in client.get_paginator("list_objects_v2").paginate(Bucket=BRONZE_BUCKET):
        llaves += [obj["Key"] for obj in pagina.get("Contents", [])]

    if not llaves:
        print("El bucket esta vacio o no contiene archivos.")
        return {
            "total_particiones": 0,
            "particiones_con_problemas": 1,
            "problemas": [{"particion": "(bucket completo)", "detalle": "bucket vacio"}],
        }

    partitions = agrupar_particiones(llaves)
    problemas = revisar_particiones(partitions)

    print("-" * 75)
    print(f"{'PARTICION (FUENTE/AÑO/MES/DIA)':<40} | ESTADO")
    print("-" * 75)
    con_problema = {p["particion"]: p["detalle"] for p in problemas}
    for partition in sorted(partitions):
        estado = f"ERROR: {con_problema[partition]}" if partition in con_problema else "OK (2 archivos)"
        print(f"{partition:<40} | {estado}")
    print("-" * 75)

    resultado = {
        "total_particiones": len(partitions),
        "particiones_con_problemas": len(problemas),
        "problemas": problemas,
    }

    if resultado["particiones_con_problemas"] == 0:
        print("\nAuditoria exitosa: no hay huecos, todas las particiones estan integras.")
    else:
        print(f"\nAuditoria con problemas: {resultado['particiones_con_problemas']} particion(es) inconsistentes.")

    return resultado


if __name__ == "__main__":
    audit_bronze_layer()
