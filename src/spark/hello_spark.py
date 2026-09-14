"""
src/spark/hello_spark.py
 
Job minimo de validacion: confirma que el Spark Master + Worker estan
arriba y aceptan trabajo, ANTES de escribir logica real de negocio.
Aisla problemas de infraestructura de problemas de codigo.
"""

from pyspark.sql import SparkSession

def main() -> None:
    spark = (
        SparkSession.builder
        .appName("hello_spark_validation")
        .getOrCreate()
    )
    
    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.sparkContext.master}")
    
    df = spark.createDataFrame(
        [(1, "loan_default_risk"), (2, "personal_finance_ml")],
        ["id", "fuente"]
    )
    df.show()
    
    spark.stop()
    print("Spark job completed successfully.")
    
if __name__ == "__main__":
    main()