import boto3
from botocore.client import Config
from collections import defaultdict

# 1. Configuración de conexión a tu MinIO local (sacado de tu docker-compose)
s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='admin',
    aws_secret_access_key='password123',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

BUCKET_NAME = 'bronze-layer'
EXPECTED_FILES = {'data.parquet', '_ingestion_metadata.json'}

def audit_bronze_layer():
    print(f"Iniciando auditoría del bucket: '{BUCKET_NAME}'...\n")
    
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
    except Exception as e:
        print(f" Error al conectar con MinIO: {e}")
        return

    if 'Contents' not in response:
        print(" El bucket está vacío o no contiene archivos.")
        return

    # 2. Agrupar archivos por partición (ej. loan_default_risk/2026/08/27)
    partitions = defaultdict(set)
    
    for obj in response['Contents']:
        key = obj['Key']
        # Separar la ruta (carpetas) del nombre del archivo
        parts = key.rsplit('/', 1)
        if len(parts) == 2:
            partition_path, file_name = parts
            partitions[partition_path].add(file_name)

    # 3. Validar consistencia
    all_healthy = True
    print("-" * 75)
    print(f"{'PARTICIÓN (AÑO/MES/DÍA)':<40} | {'ESTADO'}")
    print("-" * 75)
    
    for partition, files in sorted(partitions.items()):
        if files == EXPECTED_FILES:
            print(f"{partition:<40} |  NICE cooking (2 archivos)")
        else:
            all_healthy = False
            missing = EXPECTED_FILES - files
            extra = files - EXPECTED_FILES
            
            estado = " ERROR: "
            if missing: estado += f"Faltan: {missing}. "
            if extra: estado += f"Sobran: {extra}."
            print(f"{partition:<40} | {estado}")

    print("-" * 75)
    if all_healthy:
        print("\n Auditoría exitosa: No hay huecos. Todas las particiones están íntegras.")
    else:
        print("\n Auditoría fallida: Se encontraron inconsistencias en la capa Bronze.")

if __name__ == "__main__":
    audit_bronze_layer()