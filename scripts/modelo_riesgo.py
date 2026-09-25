"""
scripts/modelo_riesgo.py

Semana 11, Paso 2: modelo de riesgo de incumplimiento.

Ver config/gold_schema.yml para el diseno completo (que features, por
que se excluye Retirement_Age, etc.). Este archivo es la implementacion.

Decision de escalado (detalle de implementacion, no de diseno de negocio,
por eso se documenta aqui y no en gold_schema.yml): las dos fuentes NO
comparten la misma escala para conceptos "equivalentes". Por ejemplo,
Monthly_Savings (ahorro MENSUAL, loan_default_risk) vs savings_usd
(ahorro que en los datos observados llega a cientos de miles de dolares,
personal_finance_ml) pueden estar en ordenes de magnitud distintos.
Comparar dolares absolutos entre las dos fuentes no tiene sentido.

En vez de eso, estandarizamos cada feature DENTRO de su propio dataset
(z-score: "cuantas desviaciones estandar por encima/debajo del promedio
de ESE dataset"). Asi el modelo aprende de la posicion relativa de un
cliente dentro de su propia distribucion, no del monto absoluto en
dolares -- es lo que hace que la "calificacion" cruzada entre datasets
sin llave comun (ver dataCatalog.yml) sea razonable.
"""

from __future__ import annotations

import os

import pandas as pd
import s3fs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET = os.getenv("MINIO_ROOT_PASSWORD", "password123")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "silver-layer")

STORAGE_OPTIONS = {
    "key": MINIO_KEY,
    "secret": MINIO_SECRET,
    "client_kwargs": {"endpoint_url": MINIO_ENDPOINT},
}

# Ver config/gold_schema.yml, seccion modelo_riesgo.features.
# Mismo orden en las dos listas: la posicion 0 de una corresponde a la
# posicion 0 de la otra.
FEATURES_ENTRENAMIENTO = ["Debt_Amount", "Monthly_Savings"]
FEATURES_PERSONAL_FINANCE = ["loan_amount_usd", "savings_usd"]
TARGET = "Loan_Default_Risk"
UMBRAL_RIESGO_ALTO = 0.5


def _leer_particion(fuente: str, fecha: str) -> pd.DataFrame:
    ruta = f"s3://{SILVER_BUCKET}/{fuente}/{fecha}/valid"
    return pd.read_parquet(ruta, storage_options=STORAGE_OPTIONS)


def entrenar_modelo(fecha: str) -> dict:
    """Entrena el modelo de riesgo sobre la particion de loan_default_risk
    de `fecha`. Regresa el modelo entrenado y sus metricas de evaluacion.

    Se re-entrena en cada corrida (ver 'nota_limitacion' en
    gold_schema.yml: es una simplificacion valida para esta etapa del
    proyecto, no un pipeline de ML productivo con registro de modelos)."""
    df = _leer_particion("loan_default_risk", fecha)

    X = df[FEATURES_ENTRENAMIENTO]
    y = df[TARGET]

    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_escalado, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "filas_entrenamiento": int(len(X_train)),
        "filas_prueba": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, predicciones)), 3),
        "auc": (
            round(float(roc_auc_score(y_test, probabilidades)), 3)
            if y_test.nunique() > 1
            else None
        ),
        "coeficientes": dict(
            zip(FEATURES_ENTRENAMIENTO, [round(float(c), 3) for c in modelo.coef_[0]])
        ),
    }

    return {"modelo": modelo, "metricas": metricas}


def calificar_clientes(df_personal_finance: pd.DataFrame, modelo) -> pd.DataFrame:
    """Aplica el modelo ya entrenado sobre personal_finance_ml.

    Importante: se ajusta un escalador NUEVO, propio de personal_finance_ml
    (no se reutiliza el de entrenamiento, que esta calibrado a la escala
    de loan_default_risk) -- ver el docstring de este modulo."""
    X = df_personal_finance[FEATURES_PERSONAL_FINANCE].fillna(0)

    escalador_propio = StandardScaler()
    X_escalado = escalador_propio.fit_transform(X)

    probabilidades = modelo.predict_proba(X_escalado)[:, 1]

    resultado = df_personal_finance.copy()
    resultado["riesgo_score"] = probabilidades.round(4)
    resultado["riesgo_flag"] = resultado["riesgo_score"].apply(
        lambda p: "Alto" if p >= UMBRAL_RIESGO_ALTO else "Bajo"
    )
    return resultado