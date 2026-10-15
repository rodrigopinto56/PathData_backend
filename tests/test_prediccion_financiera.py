"""
tests/test_prediccion_financiera.py

Semana 15: pronostico financiero del cliente demo y estabilidad.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import datos_sinteticos as ds  # noqa: E402
import prediccion_financiera as P  # noqa: E402


@pytest.fixture(scope="module")
def serie():
    return P.serie_mensual(ds.generar_transacciones())


def test_serie_mensual_completa_y_consistente(serie):
    assert len(serie) == 60
    assert (serie["ahorro"] == (serie["ingreso"] - serie["gasto"]).round(2)).all()
    assert serie.index.is_monotonic_increasing


def test_meses_sin_movimientos_cuentan_como_cero():
    tx = pd.DataFrame({
        "fecha": ["2024-01-05", "2024-03-25"], "descripcion": ["a", "b"], "categoria": ["x", "y"],
        "monto": [100.0, 50.0], "tipo": ["Income", "Expense"],
    })
    serie = P.serie_mensual(tx)
    assert list(serie.index.astype(str)) == ["2024-01", "2024-02", "2024-03"]
    assert serie.loc[pd.Period("2024-02"), "ingreso"] == 0


def test_mes_final_incompleto_se_descarta():
    tx = pd.DataFrame({"fecha": ["2024-01-25", "2024-02-03"], "descripcion": "", "categoria": "x",
                       "monto": [10.0, 10.0], "tipo": "Expense"})
    assert list(P.serie_mensual(tx).index.astype(str)) == ["2024-01"]


def test_modelos_regresan_horizonte_completo():
    y = np.arange(30, dtype=float)
    for nombre, modelo in P.MODELOS.items():
        assert len(modelo(y, 6)) == 6, nombre


def test_holt_detecta_tendencia_y_naive_no():
    y = np.arange(40, dtype=float) * 10
    assert P.holt_amortiguado(y, 3)[0] > y[-1]
    assert P.naive(y, 3)[0] == y[-1]


def test_backtest_sin_fuga_del_futuro():
    """El error del naive a 1 paso es exactamente la diferencia con el mes anterior."""
    y = np.array([float(i % 7) for i in range(40)])
    errores = P.backtest(y, P.naive, h=1, origenes=5)
    esperados = [y[t] - y[t - 1] for t in range(35, 40)]
    np.testing.assert_allclose(errores[:, 0], esperados)


def test_estacionalidad_pura_la_gana_el_naive_estacional():
    y = np.tile([10, 20, 30, 40, 50, 60, 70, 60, 50, 40, 30, 20], 5).astype(float)
    tabla = P.comparar_modelos(y)
    assert tabla.iloc[0]["modelo"] == "naive_estacional"
    assert tabla.iloc[0]["mae"] == 0


def test_pronostico_con_banda_y_baseline_como_piso(serie):
    resultado = P.pronosticar(serie)
    futuro = resultado["pronostico"].query("tipo == 'pronostico'")
    assert len(futuro) == 3 * P.HORIZONTE
    assert (futuro["limite_inferior"] <= futuro["limite_superior"]).all()
    # Solo se usa un modelo si le gana al baseline
    for serie_nombre, elegido in resultado["modelos_elegidos"].items():
        tabla = pd.DataFrame(resultado["comparacion_modelos"][serie_nombre]).set_index("modelo")
        assert tabla.loc[elegido, "mae"] <= tabla.loc[P.BASELINE, "mae"]
    ahorro = futuro.query("serie == 'ahorro'")["valor"].to_numpy()
    ingreso = futuro.query("serie == 'ingreso'")["valor"].to_numpy()
    gasto = futuro.query("serie == 'gasto'")["valor"].to_numpy()
    np.testing.assert_allclose(ahorro, ingreso - gasto, atol=0.02)


def test_estabilidad():
    estable = pd.DataFrame({"ingreso": [100.0] * 12, "gasto": [80.0, 81.0] * 6})
    volatil = pd.DataFrame({"ingreso": [10.0, 200.0] * 6, "gasto": [80.0] * 12})
    assert P.estabilidad(estable)["estable"] is True
    resultado = P.estabilidad(volatil)
    assert resultado["estable"] is False and resultado["cv_6m_peor_pct"] == resultado["cv_ingreso_6m_pct"]
