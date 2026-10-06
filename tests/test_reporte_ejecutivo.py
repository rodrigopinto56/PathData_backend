"""
tests/test_reporte_ejecutivo.py

Pruebas de la capa de lenguaje de negocio (Semana 13): diagnostico,
acciones sugeridas, tablas por segmento y reporte HTML.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import reporte_ejecutivo as rep  # noqa: E402
from kpis_financieros import CATALOGO_KPIS, calcular_kpis_globales, calcular_kpis_por_segmento  # noqa: E402
from test_gold_kpis import FECHA, df_calificado, tablas  # noqa: E402,F401  (fixtures compartidas)


def _kpis_manual(valores: dict, estados: dict) -> pd.DataFrame:
    """Tabla de KPIs con todos los del catalogo en OK salvo lo indicado."""
    filas = []
    for kpi in CATALOGO_KPIS:
        filas.append(
            {
                "fecha": FECHA,
                "kpi": kpi,
                "valor": valores.get(kpi, 10.0),
                "estado": estados.get(kpi, "OK"),
            }
        )
    return pd.DataFrame(filas)


def _segmentos_region(indices: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fecha": FECHA, "dimension": "region", "segmento": region, "num_clientes": 100,
                "indice_riesgo_promedio": indice, "pct_semaforo_alto": 5.0, "exposicion_usd": 1e6,
                "perdida_esperada_usd": 1e5, "tasa_ahorro_mediana_pct": 25.0,
            }
            for region, indice in indices.items()
        ]
    )


@pytest.fixture
def resumen_real(tablas):
    kpis = calcular_kpis_globales(tablas, FECHA, pd.DataFrame({"Loan_Default_Risk": [0, 1]}), {"auc": 0.8})
    segmentos = calcular_kpis_por_segmento(tablas, FECHA)
    return rep.resumen_ejecutivo(kpis, segmentos), segmentos


def test_resumen_completo_con_datos_del_pipeline(resumen_real):
    resumen, _ = resumen_real
    assert [t["titulo"] for t in resumen["tarjetas"]] == [
        "Capacidad de ahorro", "Endeudamiento mensual", "Índice de riesgo", "Estabilidad de ingresos y gastos",
    ]
    assert resumen["tarjetas"][3]["estado"] == rep.SIN_DATOS
    assert sum(s["pct"] for s in resumen["semaforo"]) == pytest.approx(100, abs=0.1)
    assert sum(s["clientes"] for s in resumen["semaforo"]) == pytest.approx(resumen["total_clientes"], abs=1)
    assert resumen["diagnostico"]["nivel"] in {rep.SALUDABLE, rep.ATENCION, rep.CRITICO}
    assert resumen["acciones"]


def test_particion_vieja_sin_kpis_de_negocio_regresa_none():
    kpis = _kpis_manual({}, {})
    kpis = kpis[kpis["kpi"] != "indice_riesgo_consolidado_promedio"]
    assert rep.resumen_ejecutivo(kpis) is None


def test_todo_en_meta_es_saludable_y_solo_pide_seguimiento():
    kpis = _kpis_manual({"pct_clientes_semaforo_alto": 0.0}, {"varianza_ingresos_gastos": "SIN_DATOS"})
    resumen = rep.resumen_ejecutivo(kpis)
    assert resumen["diagnostico"]["nivel"] == rep.SALUDABLE
    assert [a["prioridad"] for a in resumen["acciones"]] == ["Seguimiento"]


def test_muchos_clientes_en_rojo_es_critico():
    kpis = _kpis_manual({"pct_clientes_semaforo_alto": 35.0}, {"pct_clientes_semaforo_alto": "ALERTA"})
    assert rep.resumen_ejecutivo(kpis)["diagnostico"]["nivel"] == rep.CRITICO


def test_endeudamiento_en_alerta_sugiere_reestructura_con_prioridad_alta():
    kpis = _kpis_manual(
        {"endeudamiento_mensual_mediana_pct": 45.0, "pct_clientes_semaforo_alto": 0.0},
        {"endeudamiento_mensual_mediana_pct": "ALERTA"},
    )
    resumen = rep.resumen_ejecutivo(kpis)
    assert resumen["diagnostico"]["nivel"] == rep.ATENCION
    assert resumen["acciones"][0]["prioridad"] == "Alta"
    assert "reestructura" in resumen["acciones"][0]["titulo"]


def test_acciones_ordenadas_por_prioridad():
    kpis = _kpis_manual(
        {"pct_clientes_semaforo_alto": 25.0},
        {"capacidad_ahorro_mediana_pct": "ALERTA", "pct_exposicion_riesgo_alto": "ALERTA", "pct_clientes_semaforo_alto": "ALERTA"},
    )
    prioridades = [a["prioridad"] for a in rep.resumen_ejecutivo(kpis, _segmentos_region({"A": 30})).get("acciones")]
    assert prioridades == sorted(prioridades, key={"Alta": 0, "Media": 1, "Seguimiento": 2}.get)


def test_foco_de_riesgo_solo_si_la_region_sobresale():
    kpis = _kpis_manual({"pct_clientes_semaforo_alto": 0.0}, {})
    parejo = rep.resumen_ejecutivo(kpis, _segmentos_region({"Norte": 30.0, "Sur": 30.5}))
    assert not any("región" in a["titulo"] for a in parejo["acciones"])

    desigual = rep.resumen_ejecutivo(kpis, _segmentos_region({"Norte": 30.0, "Sur": 45.0}))
    assert any(a["titulo"] == "Priorizar la región Sur" for a in desigual["acciones"])


def test_riesgo_por_segmento_traduce_y_ordena(resumen_real):
    _, segmentos = resumen_real
    tabla = rep.riesgo_por_segmento(segmentos, "employment_status")
    assert "Empleado" in tabla["Segmento"].tolist()
    assert tabla["Índice de riesgo"].is_monotonic_decreasing


def test_reporte_html_autocontenido_y_escapado(resumen_real):
    resumen, segmentos = resumen_real
    segmentos = segmentos.copy()
    segmentos.loc[segmentos["dimension"] == "region", "segmento"] = "<script>alert(1)</script>"
    contenido = rep.generar_reporte_html(resumen, segmentos)

    assert contenido.startswith("<!DOCTYPE html>")
    assert "Salud financiera de la cartera" in contenido
    assert "<script>" not in contenido  # texto de los datos escapado, sin JS
    assert "src=" not in contenido and "<link" not in contenido  # sin recursos externos
    for accion in resumen["acciones"]:
        assert rep.html.escape(accion["titulo"]) in contenido


def test_formatos_para_ejecutivo():
    assert rep.fmt_usd(3_246_100_000) == "$3.2 mil millones"
    assert rep.fmt_usd(1_349_900) == "$1.3 millones"
    assert rep.fmt_usd(249_798) == "$249,798"
    assert rep.fmt_pct(None) == "N/D"
    assert rep.fmt_int(32424.4) == "32,424"
