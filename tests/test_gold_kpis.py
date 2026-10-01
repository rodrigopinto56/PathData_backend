"""
tests/test_gold_kpis.py

Pruebas del modelo estrella (Semana 11) y de los KPIs financieros
(Semana 12) con datos sinteticos -- no necesitan MinIO ni Airflow.

Uso:
    pip install pandas pyarrow scikit-learn pytest
    pytest tests/
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from kpis_financieros import (  # noqa: E402
    CATALOGO_KPIS,
    intervalo_confianza_mediana,
    calcular_kpis_globales,
    calcular_kpis_por_segmento,
)
from modelo_dimensional import (  # noqa: E402
    LGD,
    construir_modelo_estrella,
    indice_riesgo_consolidado,
    pago_mensual_deuda,
    validar_integridad,
)

FECHA = "2026/09/22"


@pytest.fixture
def df_calificado() -> pd.DataFrame:
    """Imita personal_finance_ml de Silver ya calificado por el modelo."""
    rng = np.random.default_rng(42)
    n = 500
    has_loan = rng.choice(["Yes", "No"], size=n, p=[0.4, 0.6])
    riesgo_score = rng.uniform(0, 1, size=n).round(4)
    ingreso = rng.uniform(1000, 10000, size=n).round(2)
    return pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(n)],
            "age": rng.integers(18, 80, size=n),
            "gender": rng.choice(["Male", "Female", "Other"], size=n),
            "education_level": rng.choice(["High School", "Bachelor", "Master"], size=n),
            "employment_status": rng.choice(["Employed", "Self-employed", "Unemployed"], size=n),
            "region": rng.choice(["North", "South", "East", "West"], size=n),
            "monthly_income_usd": ingreso,
            "monthly_expenses_usd": (ingreso * rng.uniform(0.5, 1.2, size=n)).round(2),
            "savings_usd": rng.uniform(0, 100000, size=n).round(2),
            "has_loan": has_loan,
            "loan_type": np.where(has_loan == "Yes", rng.choice(["Home", "Car", "Personal"], size=n), None),
            "loan_amount_usd": np.where(has_loan == "Yes", rng.uniform(1000, 50000, size=n).round(2), np.nan),
            "monthly_emi_usd": np.where(has_loan == "Yes", rng.uniform(50, 1500, size=n).round(2), np.nan),
            "debt_to_income_ratio": np.where(has_loan == "Yes", rng.uniform(0, 1, size=n).round(3), 0.0),
            "credit_score": rng.integers(300, 851, size=n),
            "riesgo_score": riesgo_score,
            "riesgo_flag": np.where(riesgo_score >= 0.5, "Alto", "Bajo"),
        }
    )


@pytest.fixture
def tablas(df_calificado) -> dict[str, pd.DataFrame]:
    return construir_modelo_estrella(df_calificado, FECHA)


# ---------------------------------------------------------------------------
# Semana 11: modelo estrella
# ---------------------------------------------------------------------------
def test_modelo_estrella_tiene_todas_las_tablas(tablas):
    assert set(tablas) == {
        "dim_fecha",
        "dim_cliente",
        "dim_region",
        "dim_banda_credito",
        "dim_tipo_prestamo",
        "fact_posicion_financiera",
    }


def test_integridad_referencial(tablas):
    resultado = validar_integridad(tablas)
    assert resultado["exito_global"], [c for c in resultado["detalle"] if not c["exito"]]


def test_integridad_detecta_fk_huerfana(tablas):
    tablas["fact_posicion_financiera"].loc[0, "region_id"] = 999
    assert not validar_integridad(tablas)["exito_global"]


def test_grano_del_hecho(df_calificado, tablas):
    fact = tablas["fact_posicion_financiera"]
    assert len(fact) == len(df_calificado)
    assert fact["fecha_id"].unique().tolist() == [20260922]


def test_sin_prestamo_tiene_miembro_propio(tablas):
    dim = tablas["dim_tipo_prestamo"]
    assert "Sin prestamo" in dim["tipo_prestamo"].tolist()
    assert tablas["fact_posicion_financiera"]["tipo_prestamo_id"].notna().all()


def test_hecho_no_repite_atributos_descriptivos(tablas):
    """La normalizacion consiste justo en esto: los textos viven en las dimensiones."""
    columnas = set(tablas["fact_posicion_financiera"].columns)
    assert not columnas & {"region", "gender", "loan_type", "employment_status", "rango_credit_score"}


def test_perdida_esperada_por_cliente(tablas):
    fact = tablas["fact_posicion_financiera"]
    esperado = (fact["riesgo_score"] * fact["loan_amount_usd"] * LGD).round(2)
    pd.testing.assert_series_equal(fact["perdida_esperada_usd"], esperado, check_names=False)
    assert (fact.loc[fact["tiene_prestamo"] == 0, "perdida_esperada_usd"] == 0).all()


# ---------------------------------------------------------------------------
# Semana 12: KPIs
# ---------------------------------------------------------------------------
def test_kpis_globales_completos(tablas):
    df_loan_default = pd.DataFrame({"Loan_Default_Risk": [0, 1, 0, 1]})
    kpis = calcular_kpis_globales(tablas, FECHA, df_loan_default, {"auc": 0.81})

    assert set(kpis["kpi"]) == set(CATALOGO_KPIS)
    assert set(kpis["categoria"]) == {
        "kpis_negocio", "riesgo_financiero", "exposicion", "comportamiento_transaccional",
    }
    # El unico KPI sin datos es la varianza (no hay historial mensual)
    sin_datos = kpis[kpis["valor"].isna()]
    assert sin_datos["kpi"].tolist() == ["varianza_ingresos_gastos"]
    assert sin_datos["estado"].tolist() == ["SIN_DATOS"]
    assert set(kpis["estado"]) <= {"OK", "ALERTA", "INFO", "SIN_DATOS"}

    valores = dict(zip(kpis["kpi"], kpis["valor"]))
    assert valores["tasa_incumplimiento_historica_pct"] == 50.0
    assert valores["auc_modelo_riesgo"] == 0.81


def test_kpis_exposicion_cuadran_con_el_hecho(tablas):
    fact = tablas["fact_posicion_financiera"]
    kpis = calcular_kpis_globales(tablas, FECHA)
    valores = dict(zip(kpis["kpi"], kpis["valor"]))

    exposicion = fact.loc[fact["tiene_prestamo"] == 1, "loan_amount_usd"].sum()
    assert valores["exposicion_total_usd"] == pytest.approx(exposicion, rel=1e-6)
    assert valores["num_prestamos_activos"] == fact["tiene_prestamo"].sum()
    assert 0 < valores["indice_concentracion_region_hhi"] <= 1
    assert valores["perdida_esperada_usd"] <= valores["exposicion_total_usd"] * LGD


def test_estado_respeta_umbral(tablas):
    tablas["fact_posicion_financiera"]["es_riesgo_alto"] = 1
    kpis = calcular_kpis_globales(tablas, FECHA)
    fila = kpis[kpis["kpi"] == "pct_clientes_riesgo_alto"].iloc[0]
    assert fila["valor"] == 100.0
    assert fila["estado"] == "ALERTA"


def test_kpis_por_segmento_cuadran_con_el_total(tablas):
    segmento = calcular_kpis_por_segmento(tablas, FECHA)
    total_clientes = len(tablas["fact_posicion_financiera"])

    for dimension, grupo in segmento.groupby("dimension"):
        assert grupo["num_clientes"].sum() == total_clientes, dimension
        assert grupo["pct_exposicion_del_total"].sum() == pytest.approx(100, abs=0.1), dimension

    assert set(segmento["dimension"]) == {
        "region",
        "rango_credit_score",
        "tipo_prestamo",
        "rango_edad",
        "employment_status",
        "perfil_ahorro",
        "nivel_riesgo_consolidado",
    }


# ---------------------------------------------------------------------------
# KPIs de negocio (lamina "KPI Negocio")
# ---------------------------------------------------------------------------
def test_endeudamiento_usa_la_mensualidad_si_existe(tablas, df_calificado):
    fact = tablas["fact_posicion_financiera"]
    assert fact.attrs["fuente_pago_mensual"] == "monthly_emi_usd"
    con_prestamo = df_calificado["has_loan"] == "Yes"
    esperado = (df_calificado["monthly_emi_usd"] / df_calificado["monthly_income_usd"]).round(4)
    pd.testing.assert_series_equal(
        fact.loc[con_prestamo, "ratio_endeudamiento_mensual"], esperado[con_prestamo], check_names=False
    )
    assert (fact.loc[~con_prestamo, "ratio_endeudamiento_mensual"] == 0).all()


def test_endeudamiento_calcula_amortizacion_sin_mensualidad():
    """$12,000 a 12 meses al 12% anual -> mensualidad conocida de $1,066.19."""
    df = pd.DataFrame(
        {"loan_amount_usd": [12000.0, 1200.0], "loan_interest_rate_pct": [12.0, 0.0], "loan_term_months": [12, 12]}
    )
    pago, fuente = pago_mensual_deuda(df, pd.Series([1, 1]))
    assert fuente.startswith("amortizacion")
    assert pago.round(2).tolist() == [1066.19, 100.0]


def test_endeudamiento_sin_columnas_queda_nulo():
    pago, fuente = pago_mensual_deuda(pd.DataFrame({"loan_amount_usd": [1000.0]}), pd.Series([1]))
    assert fuente == "no_disponible"
    assert pago.isna().all()


def test_indice_consolidado_extremos():
    fact = pd.DataFrame(
        {
            # cliente sano, cliente riesgoso, cliente sin dato de endeudamiento
            "ratio_endeudamiento_mensual": [0.10, 0.50, np.nan],
            "riesgo_score": [0.0, 1.0, 1.0],
            "tasa_ahorro": [0.30, -0.10, -0.10],
            "credit_score": [800, 500, 500],
        }
    )
    indice = indice_riesgo_consolidado(fact)
    assert indice.tolist() == [0.0, 100.0, 100.0]  # el faltante no "premia" al cliente


def test_intervalo_confianza_contiene_la_mediana():
    valores = pd.Series(np.random.default_rng(0).normal(0.3, 0.1, 2000))
    inferior, superior = intervalo_confianza_mediana(valores)
    assert inferior < valores.median() < superior
    assert superior - inferior < 0.02  # con 2000 datos el intervalo es angosto


def test_kpis_de_negocio_calculados(tablas):
    kpis = calcular_kpis_globales(tablas, FECHA).set_index("kpi")
    assert kpis.loc["capacidad_ahorro_ic95_inf_pct", "valor"] <= kpis.loc["capacidad_ahorro_mediana_pct", "valor"]
    assert kpis.loc["capacidad_ahorro_mediana_pct", "valor"] <= kpis.loc["capacidad_ahorro_ic95_sup_pct", "valor"]
    semaforo = kpis.loc[["pct_clientes_semaforo_bajo", "pct_clientes_semaforo_medio", "pct_clientes_semaforo_alto"], "valor"]
    assert semaforo.sum() == pytest.approx(100, abs=0.1)
