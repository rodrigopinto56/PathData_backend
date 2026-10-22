"""
tests/test_simulador_montecarlo.py

Semana 16: simulador Monte Carlo. Se valida contra resultados conocidos
(formula cerrada de Vasicek/Basilea) y contra el sentido comun de negocio
(un escenario peor nunca da menos perdida).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import datos_sinteticos as ds  # noqa: E402
import prediccion_financiera as P  # noqa: E402
import simulador_montecarlo as M  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return M.cargar_config()


@pytest.fixture(scope="module")
def fact():
    """Tabla de hechos minima con PD heterogeneas y prestamos de distinto monto."""
    rng = np.random.default_rng(5)
    n = 1500
    tiene = (rng.random(n) < 0.6).astype(int)
    return pd.DataFrame({
        "tiene_prestamo": tiene,
        "riesgo_score": np.where(tiene == 1, rng.beta(1.2, 20, n), np.nan),
        "loan_amount_usd": np.where(tiene == 1, rng.uniform(1_000, 40_000, n), 0.0),
    })


@pytest.fixture(scope="module")
def serie():
    return P.serie_mensual(ds.generar_transacciones())


# --- Cartera -----------------------------------------------------------------
def test_correlacion_basilea_entre_3_y_16_por_ciento():
    pd_ = np.array([1e-6, 0.01, 0.05, 0.2, 0.99])
    rho = M.correlacion_basilea(pd_)
    assert np.all((rho >= 0.03 - 1e-9) & (rho <= 0.16 + 1e-9))
    assert np.all(np.diff(rho) < 0)  # a mayor PD, menor correlacion


def test_perdida_esperada_simulada_cuadra_con_la_formula(fact, config):
    """Promedio de miles de anos simulados ~= suma de PD x EAD x LGD."""
    pd_, ead = M.preparar_cartera(fact)
    escenario = config["cartera"]["escenarios"]["base"]
    resumen, _ = M.simular_escenario_cartera(pd_, ead, escenario, config["cartera"], simulaciones=20_000)
    teorica = float((pd_ * ead * escenario["lgd"]).sum())
    assert resumen["perdida_esperada"] == pytest.approx(teorica, rel=0.03)


def test_var_coincide_con_la_formula_de_vasicek():
    """Cartera grande y homogenea: el VaR simulado se acerca a la formula
    cerrada de Basilea  Phi((Phi^-1(PD) + sqrt(rho) Phi^-1(q)) / sqrt(1-rho))."""
    n, pd_0, rho = 3000, 0.02, 0.12
    perdidas = M.simular_perdidas(np.full(n, pd_0), np.ones(n), lgd=1.0, correlacion=rho,
                                  simulaciones=20_000, semilla=1)
    for q in (0.95, 0.99):
        teorica = norm.cdf((norm.ppf(pd_0) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho)) * n
        assert np.quantile(perdidas, q) == pytest.approx(teorica, rel=0.06)


def test_sin_correlacion_no_hay_colas_gordas():
    """Con rho=0 los impagos son independientes: el VaR 99% queda cerca de
    la media; con correlacion la cola se alarga mucho (por eso importa)."""
    n = 2000
    independiente = M.simular_perdidas(np.full(n, 0.02), np.ones(n), 1.0, 0.0, 10_000, 3)
    correlacionada = M.simular_perdidas(np.full(n, 0.02), np.ones(n), 1.0, 0.15, 10_000, 3)
    assert np.quantile(independiente, 0.99) < 1.5 * independiente.mean()
    assert np.quantile(correlacionada, 0.99) > 2.5 * correlacionada.mean()


def test_escenarios_ordenados_y_resumen_consistente(fact, config):
    resultado = M.simular_cartera(fact, config["cartera"])
    assert resultado["disponible"]
    resumen = resultado["resumen"].set_index("escenario")
    for columna in ["perdida_esperada", "var_99", "var_99_9", "expected_shortfall_99"]:
        assert resumen.loc["base", columna] < resumen.loc["adverso", columna] < resumen.loc["severo", columna]
    for _, fila in resumen.iterrows():
        assert fila["perdida_esperada"] <= fila["var_95"] <= fila["var_99"] <= fila["var_99_9"]
        assert fila["expected_shortfall_99"] >= fila["var_99"]
        assert fila["capital_economico"] == pytest.approx(fila["var_99_9"] - fila["perdida_esperada"])
    # El colchon del escenario base se supera ~0.1% de los anos (por definicion)
    assert resumen.loc["base", "prob_superar_colchon_pct"] <= 0.2
    assert resumen.loc["severo", "prob_superar_colchon_pct"] > resumen.loc["adverso", "prob_superar_colchon_pct"]
    # El histograma cubre todas las simulaciones de cada escenario
    distribucion = resultado["distribucion"]
    assert (distribucion.groupby("escenario")["anos"].sum() == config["cartera"]["simulaciones"]).all()


def test_reproducible_con_la_misma_semilla(fact, config):
    pd_, ead = M.preparar_cartera(fact)
    escenario = config["cartera"]["escenarios"]["adverso"]
    a, _ = M.simular_escenario_cartera(pd_, ead, escenario, config["cartera"], simulaciones=500)
    b, _ = M.simular_escenario_cartera(pd_, ead, escenario, config["cartera"], simulaciones=500)
    assert a == b


def test_cartera_sin_prestamos_no_falla(config):
    vacio = pd.DataFrame({"tiene_prestamo": [0, 0], "riesgo_score": [np.nan, np.nan], "loan_amount_usd": [0.0, 0.0]})
    assert M.simular_cartera(vacio, config["cartera"]) == {"disponible": False}


# --- Cliente demo --------------------------------------------------------------
def test_abanico_ordenado_y_probabilidades_validas(serie, config):
    resultado = M.simular_cliente(serie, config["cliente_demo"]["escenarios"]["base"], config["cliente_demo"])
    abanico = resultado["abanico"]
    assert len(abanico) == config["cliente_demo"]["horizonte_meses"]
    assert (abanico["p5"] <= abanico["p25"]).all() and (abanico["p25"] <= abanico["p50"]).all()
    assert (abanico["p50"] <= abanico["p75"]).all() and (abanico["p75"] <= abanico["p95"]).all()
    assert 0 <= resultado["prob_quedarse_sin_ahorro_pct"] <= 100
    # Quedarse sin ahorro en algun mes >= saldo negativo en cualquier mes puntual
    assert resultado["prob_quedarse_sin_ahorro_pct"] >= abanico["prob_saldo_negativo_pct"].max() - 1e-9


def test_escenarios_del_cliente_empeoran_respecto_al_base(serie, config):
    resumen = M.simular_cliente_escenarios(serie, config["cliente_demo"])["resumen"].set_index("escenario")
    for escenario in ["inflacion", "recesion", "desempleo"]:
        assert resumen.loc[escenario, "saldo_final_mediana"] < resumen.loc["base", "saldo_final_mediana"]
        assert resumen.loc[escenario, "prob_quedarse_sin_ahorro_pct"] >= resumen.loc["base", "prob_quedarse_sin_ahorro_pct"]


def test_bootstrap_conserva_el_flujo_promedio(serie, config):
    """Sin choques, el flujo mensual simulado promedia lo mismo que la historia."""
    cfg = config["cliente_demo"]
    resultado = M.simular_cliente(serie, cfg["escenarios"]["base"], cfg, simulaciones=20_000)
    historia = serie.iloc[-cfg["meses_historia"]:]
    esperado = (historia["ingreso"] - historia["gasto"]).mean()
    assert resultado["flujo_mensual_promedio"] == pytest.approx(esperado, rel=0.05, abs=5)


def test_serie_desde_pronostico_reconstruye_la_historia(serie):
    pronostico = P.pronosticar(serie)["pronostico"]
    reconstruida = M.serie_desde_pronostico(pronostico)
    assert len(reconstruida) == len(serie)
    assert np.allclose(reconstruida["ingreso"].to_numpy(), serie["ingreso"].to_numpy())
    assert np.allclose(reconstruida["gasto"].to_numpy(), serie["gasto"].to_numpy())
