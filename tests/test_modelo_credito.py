"""
tests/test_modelo_credito.py

Semanas 14-15: dataset de features y modelo predictivo de riesgo.
Datos sinteticos con el esquema real (tests/datos_sinteticos.py).
"""

from __future__ import annotations

import os
import sys

import fsspec
import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import datos_sinteticos as ds  # noqa: E402
import features as F  # noqa: E402
import modelo_credito as M  # noqa: E402


@pytest.fixture(scope="module")
def contrato():
    return F.cargar_contrato()


@pytest.fixture(scope="module")
def config():
    return M.cargar_config()


@pytest.fixture(scope="module")
def preparado(contrato):
    return F.preparar_dataset_entrenamiento(ds.generar_credit_risk(), contrato)


@pytest.fixture(scope="module")
def entrenado(preparado, config, contrato):
    return M.entrenar_modelo(
        preparado["dataset"], preparado["metadata"], config, contrato,
        df_cartera=ds.generar_cartera(), version="v_prueba",
    )


# ---------------------------------------------------------------------------
# Semana 14: dataset de features
# ---------------------------------------------------------------------------
def test_contrato_y_codigo_coinciden(contrato):
    """Cada feature 'modelo' del YAML se calcula en ambos lados (paridad)."""
    entrenamiento = F.features_entrenamiento(ds.generar_credit_risk(200), contrato)
    cartera = F.features_cartera(ds.generar_cartera(200), contrato)
    for feature in F.features_modelo(contrato):
        assert feature in entrenamiento.columns, feature
        assert feature in cartera.columns, feature


def test_atributos_protegidos_nunca_entran_al_modelo(contrato, preparado):
    seleccionadas = preparado["metadata"]["features_seleccionadas"]
    assert "edad" not in seleccionadas and "genero" not in seleccionadas
    assert set(seleccionadas) <= set(F.features_modelo(contrato))


def test_reporte_de_seleccion_explica_cada_exclusion(preparado):
    reporte = preparado["reporte_seleccion"]
    excluidas = reporte[reporte["decision"] == "excluida"]
    assert not excluidas.empty and excluidas["razon"].str.len().gt(0).all()
    assert (reporte["iv"] >= 0).all()


def test_redundancia_elimina_la_variable_de_menor_iv(preparado):
    """tasa_interes y grado_credito estan muy correlacionadas: se queda una."""
    seleccionadas = preparado["metadata"]["features_seleccionadas"]
    assert not {"tasa_interes", "grado_credito"} <= set(seleccionadas)


def test_particion_estratificada_y_reproducible(contrato, preparado):
    dataset = preparado["dataset"]
    tasas = dataset.groupby("particion_modelo")[F.TARGET].mean()
    assert tasas.max() - tasas.min() < 0.01
    otra = F.preparar_dataset_entrenamiento(ds.generar_credit_risk(), contrato)["dataset"]
    assert (otra["particion_modelo"] == dataset["particion_modelo"]).all()
    assert preparado["metadata"]["calidad"]["exito_global"]


def test_woe_y_iv_conocidos():
    """Una variable que separa perfecto tiene IV alto; una aleatoria, ~0."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 4000))
    X = pd.DataFrame({"perfecta": y * 10 + rng.normal(0, 0.1, 4000), "ruido": rng.normal(size=4000)})
    binning = F.BinningWoE().fit(X, y)
    assert binning.iv["perfecta"] > 2
    assert binning.iv["ruido"] < 0.02


def test_binning_monotono_corrige_ruido():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.uniform(0, 1, 5000))
    y = pd.Series((rng.random(5000) < 0.1 + 0.3 * x).astype(int))
    binning = F.BinningWoE(monotonia={"x": 1}).fit(pd.DataFrame({"x": x}), y)
    tasas = binning.tablas["x"].query("bin >= 0").sort_values("bin")["tasa_impago"]
    assert tasas.is_monotonic_increasing


def test_grado_por_cuantiles_respeta_distribucion():
    score = pd.Series(np.arange(1000, dtype=float))
    grado = F.grado_por_cuantiles(score, F.DISTRIBUCION_GRADOS_DEFECTO)
    assert grado.loc[999] == 1 and grado.loc[0] == 7  # mejor score -> A, peor -> G
    assert (grado == 1).mean() == pytest.approx(0.33, abs=0.01)


def test_alineacion_por_cuantiles_elimina_el_cambio_de_distribucion():
    rng = np.random.default_rng(2)
    referencia = pd.DataFrame({"r": rng.uniform(0, 0.8, 3000)})
    cartera = pd.DataFrame({"r": rng.uniform(2, 9, 3000)})  # otra escala
    alineada = F.alinear_por_cuantiles(cartera, F.cuantiles_referencia(referencia, ["r"]))
    assert F.psi(referencia["r"], cartera["r"]) > 1
    assert F.psi(referencia["r"], alineada["r"]) < 0.05
    assert alineada["r"].rank().corr(cartera["r"].rank()) > 0.999  # conserva el orden


# ---------------------------------------------------------------------------
# Semana 15: modelo
# ---------------------------------------------------------------------------
def test_modelo_tiene_poder_predictivo(entrenado):
    prueba = entrenado["card"]["metricas_prueba"]
    assert prueba["auc"] > 0.7 and prueba["gini"] == pytest.approx(2 * prueba["auc"] - 1, abs=1e-3)
    for candidato in entrenado["card"]["candidatos"].values():
        assert {"auc", "gini", "ks", "brier"} <= set(candidato["validacion_cruzada"])


def test_calibracion_respeta_la_tasa_real(entrenado):
    prueba = entrenado["card"]["metricas_prueba"]
    assert prueba["tasa_predicha"] == pytest.approx(prueba["tasa_real"], abs=0.03)


def test_ajuste_de_tasa_base_conserva_orden_y_cambia_nivel():
    p = np.array([0.05, 0.2, 0.5, 0.8])
    ajustada = M.ajustar_tasa_base(p, tasa_origen=0.22, tasa_objetivo=0.05)
    assert np.all(np.diff(ajustada) > 0) and np.all(ajustada < p)
    assert M.ajustar_tasa_base(np.array([0.22]), 0.22, 0.05)[0] == pytest.approx(0.05)


def test_pruebas_de_negocio_pasan(entrenado):
    for variable, resultado in entrenado["card"]["pruebas_monotonia"].items():
        assert resultado["exito"], variable


def test_scorecard_elegida_si_el_retador_no_gana_por_margen(entrenado):
    card = entrenado["card"]
    sc, bo = card["candidatos"]["scorecard"]["prueba"]["auc"], card["candidatos"]["boosting"]["prueba"]["auc"]
    esperado = "boosting" if bo - sc >= 0.02 and card["candidatos"]["boosting"]["prueba"]["brier"] <= card["candidatos"]["scorecard"]["prueba"]["brier"] else "scorecard"
    assert card["modelo_elegido"] == esperado


def test_calificacion_de_cartera(entrenado, contrato):
    cartera = ds.generar_cartera()
    calificada = entrenado["artefacto"].calificar_cartera(cartera, contrato)
    con_prestamo = calificada["has_loan"] == "Yes"
    assert calificada.loc[con_prestamo, "riesgo_score"].between(0, 1).all()
    assert calificada.loc[~con_prestamo, "riesgo_score"].isna().all()
    assert set(calificada.loc[~con_prestamo, "riesgo_flag"]) == {"Sin préstamo"}
    assert set(calificada.loc[con_prestamo, "riesgo_flag"]) <= {"Alto", "Bajo"}
    # Escala de cartera (~5%), no la del entrenamiento (~23%)
    assert calificada["riesgo_score"].mean() < 0.15
    # El riesgo sube con el grado (proxy del credit score)
    por_grado = calificada.groupby("grado_credito_proxy")["riesgo_score"].mean()
    assert por_grado.loc["A"] < por_grado.loc["C"] < por_grado.loc["E"]
    # Mas puntos = menos riesgo
    assert calificada[["puntaje_riesgo", "riesgo_score"]].corr(method="spearman").iloc[0, 1] < -0.95
    assert calificada.loc[con_prestamo, "razones_riesgo"].str.len().gt(0).all()


def test_ficha_del_modelo_documenta_limites_y_equidad(entrenado):
    card = entrenado["card"]
    assert card["limitaciones"] and card["equidad"]["genero_en_cartera"]["ratio_aprobacion"] > 0
    assert card["evaluacion_cartera"]["psi_sin_alinear"]


def test_registro_campeon_retador(entrenado, preparado, config, contrato, tmp_path):
    registro = M.RegistroModelos(fsspec.filesystem("file"), str(tmp_path / "riesgo_credito"))
    assert registro.campeon() == (None, None)

    card = entrenado["card"]
    registro.guardar(card["version"], entrenado["artefacto"], card, entrenado["tabla_puntos"])
    registro.promover(card["version"], card["razon_promocion"])
    campeon, card_cargada = registro.campeon()
    assert card_cargada["version"] == "v_prueba"
    prueba = preparado["dataset"].query("particion_modelo == 'prueba'")
    np.testing.assert_allclose(campeon.pd_calibrada(prueba), entrenado["artefacto"].pd_calibrada(prueba))

    # Un retador igual de bueno (mismos datos) se promueve frente al campeon vigente
    nuevo = M.entrenar_modelo(preparado["dataset"], preparado["metadata"], config, contrato, campeon, version="v_retador")
    assert nuevo["promover"] and "campeon v_prueba" in nuevo["card"]["razon_promocion"]


def test_retador_peor_no_reemplaza_al_campeon(entrenado, preparado, config, contrato):
    """Si el nuevo entrenamiento es claramente peor, se conserva el campeon."""
    malo = preparado["dataset"].copy()
    rng = np.random.default_rng(5)
    malo[F.TARGET] = rng.permutation(malo[F.TARGET].to_numpy())  # etiquetas sin relacion
    resultado = M.entrenar_modelo(malo, preparado["metadata"], config, contrato, entrenado["artefacto"], version="v_malo")
    assert not resultado["promover"]


def test_config_del_modelo_es_coherente(config):
    assert 0 < config["calibracion"]["tasa_impago_objetivo"] < 0.5
    assert config["seleccion_campeon"]["ventaja_minima_auc_retador"] > 0
    with open(os.path.join(os.path.dirname(__file__), "..", "config", "features.yml")) as f:
        assert yaml.safe_load(f)["particion"]["semilla"] == 42
