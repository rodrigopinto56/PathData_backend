"""
tests/test_observabilidad.py

Semana 18: metricas operativas, SLA y alertas con corridas de Airflow
sinteticas (mismas columnas que exporta monitoreo_dag).
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

import observabilidad as O  # noqa: E402

ZONA = "America/Mexico_City"
AHORA = pd.Timestamp("2026-11-10 12:00", tz=ZONA)


def _corrida(dag, dia, estado="success", inicio_h=0.1, duracion_min=8.0, run_type="scheduled"):
    """Corrida programada de la particion `dia` (corre el dia siguiente a las 00:00 MX)."""
    logica = pd.Timestamp(dia, tz=ZONA)
    inicio = logica + pd.Timedelta(days=1, hours=inicio_h)
    return {"dag_id": dag, "run_id": f"{run_type}__{dia}", "run_type": run_type, "state": estado,
            "logical_date": logica.tz_convert("UTC"), "start_date": inicio.tz_convert("UTC"),
            "end_date": (inicio + pd.Timedelta(minutes=duracion_min)).tz_convert("UTC")}


@pytest.fixture(scope="module")
def config():
    return O.cargar_config()


@pytest.fixture
def corridas_sanas():
    filas = []
    for dia in pd.date_range("2026-11-01", "2026-11-09"):
        d = dia.strftime("%Y-%m-%d")
        filas += [_corrida("ingesta_bronze_dag", d, duracion_min=1), _corrida("transformacion_silver_dag", d, duracion_min=4),
                  _corrida("transformacion_gold_dag", d, inicio_h=0.2, duracion_min=6)]
    return O.preparar_corridas(pd.DataFrame(filas), ZONA)


def _entradas(corridas, **extra):
    calidad = pd.DataFrame({"fuente": ["credit_risk", "personal_finance_ml"], "fecha": ["2026/11/09"] * 2,
                            "porcentaje_calidad": [100.0, 98.0]})
    log = pd.DataFrame({"timestamp": [(AHORA - pd.Timedelta(hours=h)).isoformat() for h in range(10)],
                        "modo": ["ia"] * 9 + ["respaldo"], "segundos": [20.0] * 10, "reintentos": [0] * 10})
    base = {"corridas": corridas, "calidad": calidad, "asistente_log": log,
            "modelo": {"fecha_entrenamiento": (AHORA - pd.Timedelta(days=3)).isoformat()},
            "evaluacion": {"exito_final_pct": 86.7}, "eventos": pd.DataFrame()}
    return {**base, **extra}


# --- Metricas -----------------------------------------------------------------
def test_resumen_por_dag(corridas_sanas):
    resumen = O.resumen_por_dag(corridas_sanas).set_index("dag_id")
    assert resumen.loc["transformacion_gold_dag", "tasa_exito_pct"] == 100
    assert resumen.loc["transformacion_gold_dag", "duracion_mediana_min"] == pytest.approx(6)
    assert resumen.loc["ingesta_bronze_dag", "corridas"] == 9
    assert resumen.loc["ingesta_bronze_dag", "en_curso"] == 0


def test_frescura_a_tiempo_y_tarde(corridas_sanas):
    frescura = O.frescura_por_dia(corridas_sanas, "transformacion_gold_dag", "07:00", AHORA, 7)
    assert len(frescura) == 7 and frescura["a_tiempo"].all()
    # La particion del 9 termina a las 09:00 del dia 10: tarde por 2 horas
    tarde = corridas_sanas.copy()
    fila = (tarde["dag_id"] == "transformacion_gold_dag") & (tarde["particion"] == pd.Timestamp("2026-11-09", tz=ZONA))
    tarde.loc[fila, "end_date"] = pd.Timestamp("2026-11-10 09:00", tz=ZONA)
    resultado = O.frescura_por_dia(tarde, "transformacion_gold_dag", "07:00", AHORA, 7).set_index("particion")
    dia9 = resultado.loc[pd.Timestamp("2026-11-09", tz=ZONA)]
    assert not dia9["a_tiempo"] and dia9["retraso_min"] == pytest.approx(120)


def test_dia_sin_corrida_cuenta_como_tarde(corridas_sanas):
    sin_dia = corridas_sanas[~((corridas_sanas["dag_id"] == "transformacion_gold_dag")
                               & (corridas_sanas["particion"] == pd.Timestamp("2026-11-05", tz=ZONA)))]
    frescura = O.frescura_por_dia(sin_dia, "transformacion_gold_dag", "07:00", AHORA, 7)
    assert (~frescura["a_tiempo"]).sum() == 1


def test_frescura_no_castiga_dias_antes_del_proyecto(corridas_sanas):
    frescura = O.frescura_por_dia(corridas_sanas, "transformacion_gold_dag", "07:00", AHORA, 30)
    assert frescura["particion"].min() == pd.Timestamp("2026-11-01", tz=ZONA)


# --- SLA ------------------------------------------------------------------------
def test_todo_sano_cumple_todos_los_slas(corridas_sanas, config):
    slas = O.evaluar_slas(_entradas(corridas_sanas), config, AHORA).set_index("sla_id")
    assert (slas["estado"] == O.OK).all(), slas[slas["estado"] != O.OK]
    assert slas.loc["asistente_ia", "valor"] == 90.0


def test_fallas_y_lentitud_rompen_los_slas(corridas_sanas, config):
    malas = corridas_sanas.copy()
    gold = malas["dag_id"] == "transformacion_gold_dag"
    malas.loc[gold, "end_date"] = malas.loc[gold, "start_date"] + pd.Timedelta(minutes=40)
    malas["duracion_min"] = (malas["end_date"] - malas["start_date"]).dt.total_seconds() / 60
    malas.loc[malas.index[-4:], "state"] = "failed"  # recientes: dentro de la ventana de 7 dias
    slas = O.evaluar_slas(_entradas(malas, evaluacion={"exito_final_pct": 80.0}), config, AHORA).set_index("sla_id")
    assert slas.loc["duracion_gold", "estado"] == O.ALERTA
    assert slas.loc["exito_corridas", "estado"] == O.ALERTA
    assert slas.loc["asistente_evaluacion", "estado"] == O.ALERTA


def test_sin_datos_no_es_alerta(config):
    slas = O.evaluar_slas({"corridas": O.preparar_corridas(pd.DataFrame(), ZONA)}, config, AHORA)
    assert (slas["estado"] == O.SIN_DATOS).all()
    assert O.estado_general(slas, O.alertas_activas(slas, pd.DataFrame(), pd.DataFrame(), AHORA)) == "Sin datos"


def test_asistente_con_pocas_preguntas_queda_sin_datos(corridas_sanas, config):
    pocas = pd.DataFrame({"timestamp": [AHORA.isoformat()] * 2, "modo": ["respaldo"] * 2,
                          "segundos": [90.0] * 2, "reintentos": [1, 1]})
    slas = O.evaluar_slas(_entradas(corridas_sanas, asistente_log=pocas), config, AHORA).set_index("sla_id")
    assert slas.loc["asistente_ia", "estado"] == O.SIN_DATOS and "menos de 5" in slas.loc["asistente_ia", "detalle"]


def test_modelo_viejo_genera_alerta(corridas_sanas, config):
    viejo = {"fecha_entrenamiento": (AHORA - pd.Timedelta(days=30)).isoformat()}
    slas = O.evaluar_slas(_entradas(corridas_sanas, modelo=viejo), config, AHORA).set_index("sla_id")
    assert slas.loc["modelo_vigente", "estado"] == O.ALERTA


# --- Alertas --------------------------------------------------------------------
def test_falla_resuelta_no_es_alerta_y_abierta_si(corridas_sanas, config):
    dia = "2026-11-09"
    fallida = _corrida("transformacion_gold_dag", dia, estado="failed", inicio_h=0.1)
    con_falla = O.preparar_corridas(pd.concat([
        corridas_sanas.assign(**{c: corridas_sanas[c].dt.tz_convert("UTC") for c in ["logical_date", "start_date", "end_date"]})
        .drop(columns=["duracion_min", "particion"]), pd.DataFrame([fallida])]), ZONA)
    # Hay una exitosa de la misma particion DESPUES de la falla (inicio 0.2 h > 0.1 h) -> resuelta
    assert O.fallas_abiertas(con_falla, AHORA).empty
    sin_reintento = con_falla[~((con_falla["dag_id"] == "transformacion_gold_dag") & (con_falla["state"] == "success")
                                & (con_falla["particion"] == pd.Timestamp(dia, tz=ZONA)))]
    abiertas = O.fallas_abiertas(sin_reintento, AHORA)
    assert len(abiertas) == 1 and abiertas.iloc[0]["dag_id"] == "transformacion_gold_dag"


def test_alertas_ordenadas_y_estado_general(corridas_sanas, config):
    eventos = pd.DataFrame([{"nivel": "ALERTA", "tipo": "falla_tarea", "dag_id": "transformacion_gold_dag",
                             "task_id": "construir_gold", "error": "FileNotFoundError: silver-layer/...",
                             "timestamp": (AHORA - pd.Timedelta(hours=2)).isoformat()},
                            {"nivel": "INFO", "task": "x", "timestamp": AHORA.isoformat()}])
    entradas = _entradas(corridas_sanas, modelo={"fecha_entrenamiento": (AHORA - pd.Timedelta(days=30)).isoformat()})
    slas = O.evaluar_slas(entradas, config, AHORA)
    alertas = O.alertas_activas(slas, corridas_sanas, eventos, AHORA)
    assert list(alertas["severidad"]) == ["alta", "media"]  # tarea fallida primero, luego modelo viejo
    assert "construir_gold" in alertas.iloc[0]["elemento"]
    assert O.estado_general(slas, alertas) == "Crítico"
    assert O.estado_general(slas, alertas[alertas["severidad"] == "media"]) == "Requiere atención"


# --- Archivos -------------------------------------------------------------------
def test_registro_de_preguntas_y_lectura(tmp_path):
    O.registrar_pregunta({"timestamp": AHORA.isoformat(), "modo": "ia", "segundos": 12.0}, base=str(tmp_path))
    with open(tmp_path / "asistente_log.jsonl", "a") as f:
        f.write('{"linea": "a medio escrib')  # linea corrupta: se ignora
    entradas = O.cargar_entradas_locales(str(tmp_path))
    assert len(entradas["asistente_log"]) == 1 and entradas["corridas"].empty and entradas["evaluacion"] == {}


# --- bronzeAudit (versionado en S18) ---------------------------------------------
def test_bronze_audit_detecta_particiones_incompletas(monkeypatch):
    import types

    # bronzeAudit importa el hook de S3 de Airflow; para probar su logica pura basta un sustituto
    for nombre in ["airflow", "airflow.providers", "airflow.providers.amazon", "airflow.providers.amazon.aws",
                   "airflow.providers.amazon.aws.hooks"]:
        monkeypatch.setitem(sys.modules, nombre, types.ModuleType(nombre))
    s3 = types.ModuleType("airflow.providers.amazon.aws.hooks.s3")
    s3.S3Hook = object
    monkeypatch.setitem(sys.modules, "airflow.providers.amazon.aws.hooks.s3", s3)
    import importlib

    import bronzeAudit as B
    importlib.reload(B)

    llaves = ["loan/2026/11/01/data.parquet", "loan/2026/11/01/_ingestion_metadata.json",
              "loan/2026/11/02/data.parquet", "loan/2026/11/03/data.parquet", "loan/2026/11/03/_ingestion_metadata.json",
              "loan/2026/11/03/extra.csv"]
    problemas = {p["particion"]: p["detalle"] for p in B.revisar_particiones(B.agrupar_particiones(llaves))}
    assert set(problemas) == {"loan/2026/11/02", "loan/2026/11/03"}
    assert "faltan" in problemas["loan/2026/11/02"] and "sobran" in problemas["loan/2026/11/03"]


def test_corrida_reejecutada_no_da_duracion_negativa():
    fila = _corrida("entrenamiento_modelo_dag", "2026-11-01")
    fila["end_date"] = fila["start_date"] - pd.Timedelta(days=2)  # Clear: inicio nuevo, fin viejo
    corridas = O.preparar_corridas(pd.DataFrame([fila]), ZONA)
    assert corridas["duracion_min"].isna().all()


def test_alerta_dice_en_que_tarea_fallo(corridas_sanas, config):
    fallida = _corrida("transformacion_silver_dag", "2026-11-09", estado="failed")
    corridas = O.preparar_corridas(pd.DataFrame([fallida]), ZONA)
    tareas = pd.DataFrame([
        {"dag_id": "transformacion_silver_dag", "run_id": fallida["run_id"], "task_id": "limpiar_credit_risk", "state": "failed"},
        {"dag_id": "transformacion_silver_dag", "run_id": fallida["run_id"], "task_id": "validar_calidad__2", "state": "upstream_failed"},
    ])
    slas = O.evaluar_slas(_entradas(corridas), config, AHORA)
    alertas = O.alertas_activas(slas, corridas, pd.DataFrame(), AHORA, tareas=tareas)
    mensaje = alertas[alertas["origen"] == "Corrida fallida"].iloc[0]["mensaje"]
    assert "fallo en: limpiar_credit_risk" in mensaje and "validar_calidad" not in mensaje
