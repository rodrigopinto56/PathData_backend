"""
tests/datos_sinteticos.py

Generadores de datos sinteticos con el MISMO esquema que las fuentes
reales, para probar el pipeline de IA (Semanas 14-15) sin descargar nada:

  - generar_credit_risk():   imita kaggle laotse/credit-risk-dataset
                             (prestamos etiquetados con loan_status 0/1),
                             incluyendo sus defectos conocidos: edades
                             imposibles, antiguedad laboral de 100+ anios,
                             ~10% de tasas nulas y filas duplicadas.
  - generar_cartera():       imita personal_finance_ml en Silver.
  - generar_transacciones(): imita las transacciones del cliente demo
                             (ya con columnas normalizadas de Silver).

Las relaciones entre variables son realistas (peor grado / mas % del
ingreso comprometido -> mas impago) para que el modelo tenga senal que
aprender, pero los NUMEROS NO son los del dataset real: las metricas
que salen de aqui solo sirven para probar que el codigo funciona.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GRADOS = ["A", "B", "C", "D", "E", "F", "G"]
PROPORCION_GRADOS = [0.33, 0.32, 0.20, 0.11, 0.03, 0.007, 0.003]
TASA_POR_GRADO = {"A": 7.3, "B": 11.0, "C": 13.5, "D": 15.4, "E": 17.0, "F": 18.6, "G": 20.2}
PROPOSITOS = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]


def generar_credit_risk(n: int = 6000, semilla: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)

    # Grado de credito a partir de una "solvencia" latente (A = mejor)
    solvencia = rng.normal(size=n)
    posicion = 1 - pd.Series(solvencia).rank(pct=True).to_numpy()  # 0 = el mas solvente
    indice_grado = np.searchsorted(np.cumsum(PROPORCION_GRADOS)[:-1], posicion)
    grado = np.array(GRADOS)[indice_grado]

    ingreso = np.round(np.exp(rng.normal(np.log(55_000), 0.55, n)), 0)
    monto = np.clip(np.round(np.exp(rng.normal(np.log(8_000), 0.7, n)), -2), 500, 35_000)
    porcentaje = np.clip(np.round(monto / ingreso, 2), 0.0, 0.83)
    tasa = np.round(np.array([TASA_POR_GRADO[g] for g in grado]) + rng.normal(0, 1.2, n), 2)
    tasa[rng.random(n) < 0.10] = np.nan

    proposito = rng.choice(PROPOSITOS, n)
    vivienda = rng.choice(["RENT", "MORTGAGE", "OWN", "OTHER"], n, p=[0.5, 0.41, 0.08, 0.01])
    edad = np.clip(np.round(rng.gamma(9, 3.1, n)), 20, 75)
    antiguedad = np.clip(np.round(rng.exponential(4.5, n)), 0, 40)
    antiguedad[rng.random(n) < 0.03] = np.nan
    impago_previo = np.where(rng.random(n) < 0.12 + 0.12 * (indice_grado >= 3), "Y", "N")

    logit = (
        -3.2
        + 0.85 * indice_grado
        + 5.0 * (porcentaje - 0.17)
        - 0.45 * np.log(ingreso / 55_000)
        + 0.55 * (vivienda == "RENT")
        + 0.35 * np.isin(proposito, ["MEDICAL", "DEBTCONSOLIDATION"])
        - 0.35 * (proposito == "VENTURE")
        + rng.normal(0, 0.6, n)
    )
    estatus = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "person_age": edad.astype(int),
            "person_income": ingreso.astype(int),
            "person_home_ownership": vivienda,
            "person_emp_length": antiguedad,
            "loan_intent": proposito,
            "loan_grade": grado,
            "loan_amnt": monto.astype(int),
            "loan_int_rate": tasa,
            "loan_status": estatus,
            "loan_percent_income": porcentaje,
            "cb_person_default_on_file": impago_previo,
            "cb_person_cred_hist_length": np.clip(np.round(edad - 20 + rng.normal(0, 2, n)), 2, 30).astype(int),
        }
    )
    # Defectos conocidos del dataset real (deben ir a cuarentena / dedup)
    df.loc[0, "person_age"] = 144
    df.loc[1, "person_emp_length"] = 123
    return pd.concat([df, df.iloc[2:12]], ignore_index=True)


def generar_cartera(n: int = 3000, semilla: int = 11, con_mensualidad: bool = True) -> pd.DataFrame:
    """personal_finance_ml (Silver), sin etiqueta de impago."""
    rng = np.random.default_rng(semilla)
    tiene = rng.random(n) < 0.4
    ingreso = np.round(np.exp(rng.normal(np.log(4_000), 0.5, n)), 2)
    monto = np.where(tiene, np.round(np.exp(rng.normal(np.log(20_000), 0.8, n)), 2), np.nan)
    plazo = np.where(tiene, rng.choice([12, 24, 36, 48, 60], n), np.nan)
    tasa = np.where(tiene, np.round(rng.uniform(5, 22, n), 2), np.nan)
    df = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "age": rng.integers(18, 75, n),
            "gender": rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04]),
            "employment_status": rng.choice(["Employed", "Self-employed", "Unemployed", "Student"], n),
            "region": rng.choice(["North", "South", "East", "West", "Central"], n),
            "monthly_income_usd": ingreso,
            "monthly_expenses_usd": np.round(ingreso * rng.uniform(0.4, 1.1, n), 2),
            "savings_usd": np.round(rng.uniform(0, 150_000, n), 2),
            "has_loan": np.where(tiene, "Yes", "No"),
            "loan_type": np.where(tiene, rng.choice(["Home", "Car", "Personal", "Education", "Business", "Medical"], n), None),
            "loan_amount_usd": monto,
            "loan_term_months": plazo,
            "loan_interest_rate_pct": tasa,
            "debt_to_income_ratio": np.where(tiene, np.round(rng.uniform(0, 3, n), 3), 0.0),
            "credit_score": rng.integers(300, 851, n),
        }
    )
    if con_mensualidad:
        r = tasa / 100 / 12
        df["monthly_emi_usd"] = np.round(monto * r / (1 - (1 + r) ** -plazo), 2)
    return df


def generar_transacciones(meses: int = 60, semilla: int = 3) -> pd.DataFrame:
    """Transacciones del cliente demo, columnas ya normalizadas (Silver)."""
    rng = np.random.default_rng(semilla)
    filas = []
    for m, inicio in enumerate(pd.date_range("2020-01-01", periods=meses, freq="MS")):
        for _ in range(rng.integers(15, 30)):
            dia = inicio + pd.Timedelta(days=int(rng.integers(0, 28)))
            filas.append((dia, "Gasto", rng.choice(["Rent", "Food & Drink", "Utilities", "Shopping"]),
                          round(float(rng.uniform(100, 1500) * (1 + 0.004 * m)), 2), "Expense"))
        for _ in range(rng.integers(3, 7)):
            dia = inicio + pd.Timedelta(days=int(rng.integers(0, 28)))
            filas.append((dia, "Ingreso", rng.choice(["Salary", "Investment", "Other"]),
                          round(float(rng.uniform(1500, 4500)), 2), "Income"))
    return pd.DataFrame(filas, columns=["fecha", "descripcion", "categoria", "monto", "tipo"])
