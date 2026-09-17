# Dockerfile.airflow
#
# Imagen personalizada de Airflow con las dependencias YA instaladas.
# Reemplaza el uso de _PIP_ADDITIONAL_REQUIREMENTS (fragil: reinstala en
# cada arranque del contenedor) por una instalacion unica al construir
# la imagen -- tal como recomienda la documentacion oficial de Airflow.

FROM apache/airflow:2.9.3

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
USER airflow

RUN pip install --no-cache-dir \
    apache-airflow==2.9.3 \
    pandas \
    pyarrow \
    apache-airflow-providers-amazon \
    apache-airflow-providers-apache-spark==5.2.1 \
    pyspark==4.0.0 \
    s3fs \
    great_expectations==0.18.19