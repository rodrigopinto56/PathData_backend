"""
tests/test_asesor_cliente.py

Semana 19: asesor financiero para el cliente. Sin Ollama: el LLM se
reemplaza por un modelo falso de LangChain.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import asesor_cliente as C  # noqa: E402
import kpis_financieros as K  # noqa: E402
from test_gold_kpis import FECHA, df_calificado, tablas  # noqa: E402,F401  (fixtures compartidas)


@pytest.fixture
def clientes(tablas) -> pd.DataFrame:  # noqa: F811  (fixture de pytest)
    """Lo mismo que arma data_loader.cargar_clientes_particion."""
    return (
        tablas["fact_posicion_financiera"]
        .merge(tablas["dim_region"], on="region_id", how="left")
        .merge(tablas["dim_cliente"][["user_id", "age", "rango_edad", "employment_status"]], on="user_id", how="left")
        .merge(tablas["dim_tipo_prestamo"], on="tipo_prestamo_id", how="left")
    )


def _cliente(**cambios) -> dict:
    """Perfil armado a mano para probar reglas con numeros redondos."""
    base = {"user_id": "U00001", "edad": 30, "rango_edad": "25-34", "region": "Norte", "empleo": "Empleado",
            "ingreso": 3000.0, "gasto": 2400.0, "ahorro_acumulado": 9000.0, "tiene_prestamo": True,
            "tipo_prestamo": "Automotriz", "monto_prestamo": 20000.0, "pago_deuda": 600.0, "tasa_interes": 12.0,
            "credit_score": 700.0}
    base.update(cambios)
    base["flujo"] = base["ingreso"] - base["gasto"]
    base["tasa_ahorro"] = base["flujo"] / base["ingreso"]
    base["meses_fondo"] = base["ahorro_acumulado"] / base["gasto"]
    base["endeudamiento"] = base["pago_deuda"] / base["ingreso"] if base["tiene_prestamo"] else 0.0
    return base


def _datos(perfil, clientes_df=None) -> C.DatosCliente:
    return C.DatosCliente(fecha=FECHA, perfil=perfil, clientes=clientes_df if clientes_df is not None else pd.DataFrame())


# --- Reglas -------------------------------------------------------------------
def test_metas_coinciden_con_los_kpis_del_banco():
    assert C.UMBRAL_ENDEUDAMIENTO == K.UMBRAL_ENDEUDAMIENTO
    assert C.MESES_FONDO_MINIMO == K.MESES_COLCHON_MINIMO
    assert C.UMBRAL_AHORRO == K.UMBRAL_AHORRADOR


def test_perfil_desde_las_tablas_gold(clientes):
    uid = clientes["user_id"].iloc[0]
    p = C.perfil_cliente(clientes, uid.lower())  # sin importar mayusculas
    assert p["user_id"] == uid
    fila = clientes.iloc[0]
    assert p["ingreso"] == pytest.approx(fila["monthly_income_usd"])
    assert p["flujo"] == pytest.approx(fila["monthly_income_usd"] - fila["monthly_expenses_usd"])
    assert C.perfil_cliente(clientes, "U99999") is None
    # el cliente no ve las herramientas internas de riesgo
    assert not {"riesgo_score", "indice_riesgo_consolidado", "es_riesgo_alto"} & set(p)


def test_perfil_con_prestamo_trae_tipo_y_mensualidad(clientes):
    fila = clientes[clientes["tiene_prestamo"] == 1].iloc[0]
    p = C.perfil_cliente(clientes, fila["user_id"])
    assert p["tiene_prestamo"] and p["tipo_prestamo"] and p["pago_deuda"] > 0
    assert p["endeudamiento"] == pytest.approx(p["pago_deuda"] / p["ingreso"])


@pytest.mark.parametrize("cambios, clave, esperado", [
    ({"gasto": 2400.0}, "ahorro", "bien"),          # 20% exacto
    ({"gasto": 2700.0}, "ahorro", "atencion"),
    ({"gasto": 3300.0}, "ahorro", "urgente"),       # gasta mas de lo que gana
    ({"ahorro_acumulado": 7200.0}, "fondo", "bien"),  # 3 meses exactos
    ({"ahorro_acumulado": 3000.0}, "fondo", "atencion"),
    ({"ahorro_acumulado": 500.0}, "fondo", "urgente"),
    ({"pago_deuda": 1080.0}, "deuda", "bien"),      # 36% exacto
    ({"pago_deuda": 1400.0}, "deuda", "atencion"),
    ({"pago_deuda": 1800.0}, "deuda", "urgente"),
    ({"tiene_prestamo": False, "pago_deuda": 0.0}, "deuda", "bien"),
    ({"credit_score": 600.0}, "credito", "atencion"),
    ({"credit_score": 500.0}, "credito", "urgente"),
])
def test_semaforo_de_pilares(cambios, clave, esperado):
    pilar = next(x for x in C.pilares(_cliente(**cambios)) if x.clave == clave)
    assert pilar.estado == esperado


def test_puntaje_de_salud():
    assert C.puntaje_salud(C.pilares(_cliente())) == (100, "Sólida")
    malo = _cliente(gasto=3300.0, ahorro_acumulado=500.0, pago_deuda=1800.0, credit_score=500.0)
    assert C.puntaje_salud(C.pilares(malo)) == (20, "Frágil")


def test_plan_de_accion_con_montos_y_prioridad():
    p = _cliente(gasto=3300.0, ahorro_acumulado=500.0, pago_deuda=1800.0)
    acciones = C.plan_accion(p)
    assert [a["pilar"] for a in acciones][:3] == ["ahorro", "deuda", "fondo"]  # urgentes, en orden
    assert "$300" in acciones[0]["detalle"]                                     # le faltan 300 al mes
    assert "$1,080" in acciones[1]["detalle"] and acciones[1]["requiere_ejecutivo"]
    assert "$9,900" in acciones[2]["detalle"]                                   # 3 meses de 3,300


def test_plan_calcula_meses_para_el_fondo():
    p = _cliente(ahorro_acumulado=3000.0)  # le faltan 4,200; ahorra 600 al mes
    fondo = next(a for a in C.plan_accion(p) if a["pilar"] == "fondo")
    assert "7 meses" in fondo["detalle"]


def test_cliente_sano_recibe_mensaje_de_mantenimiento():
    acciones = C.plan_accion(_cliente(ahorro_acumulado=20000.0))
    assert len(acciones) == 1 and "buen camino" in acciones[0]["titulo"]


def test_simular_recorte():
    s = C.simular_recorte(_cliente(ahorro_acumulado=3000.0), 10)
    assert s["gasto"] == pytest.approx(2160) and s["flujo"] == pytest.approx(840)
    assert s["meses_para_fondo"] == 5  # (6,480 - 3,000) / 840 = 4.1 -> 5


def test_comparativo_solo_con_grupos_grandes(clientes):
    p = C.perfil_cliente(clientes, clientes["user_id"].iloc[0])
    c = C.comparativo(p, clientes)
    assert c["personas"] >= C.TAMANO_MINIMO_GRUPO
    assert all(0 <= f["mejor_que_pct"] <= 100 for f in c["filas"] if f["mejor_que_pct"] is not None)
    assert C.comparativo(p, clientes.head(10)) is None  # grupo chico -> no se compara


def test_clientes_ejemplo_existen(clientes):
    ejemplos = C.clientes_ejemplo(clientes)
    assert ejemplos and set(ejemplos.values()) <= set(clientes["user_id"])


# --- Enrutador ------------------------------------------------------------------
@pytest.mark.parametrize("pregunta, tema", [
    ("¿Cómo están mis finanzas?", "resumen"),
    ("¿Cuánto debería ahorrar al mes?", "ahorro"),
    ("¿Me alcanza para un fondo de emergencia?", "fondo"),
    ("¿Puedo pedir otro préstamo?", "capacidad"),
    ("¿Me aprueban un crédito para un auto nuevo?", "capacidad"),
    ("¿Cómo mejoro mi credit score?", "credito"),
    ("¿Cómo organizo mi presupuesto?", "presupuesto"),
    ("¿Estoy bien comparado con gente de mi edad?", "comparativo"),
    ("¿Me conviene invertir en bitcoin?", "inversion"),
    ("Me despidieron, ¿qué hago con mi préstamo?", "dificultad"),
    ("¿Cuánto debo de mi préstamo?", "deuda"),
])
def test_enrutador(pregunta, tema):
    ruta = C.enrutar_cliente(pregunta, "U00001")
    assert ruta["derivar"] is None and tema in ruta["temas"]


def test_fondo_de_inversion_no_es_fondo_de_emergencia():
    assert C.enrutar_cliente("¿Qué fondo de inversión me recomiendas?", "U00001")["temas"] == ["inversion"]


@pytest.mark.parametrize("pregunta, motivo", [
    ("Me robaron la tarjeta", "seguridad"),
    ("Tengo un cargo que no reconozco", "seguridad"),
    ("¿Te paso mi NIP para revisar?", "seguridad"),
    ("¿Cuánto gana el cliente U00002?", "privacidad"),
])
def test_temas_sensibles_no_pasan_por_la_ia(pregunta, motivo):
    llm = FakeListChatModel(responses=["no deberia usarse"])
    r = C.responder_cliente(pregunta, _datos(_cliente()), llm)
    assert r["modo"] == "derivado" and r["temas"] == [motivo] and "no deberia" not in r["respuesta"]


def test_su_propio_id_si_se_permite():
    assert C.enrutar_cliente("Soy U00001, ¿cómo voy?", "U00001")["derivar"] is None


# --- Respuestas -------------------------------------------------------------------
def test_modo_basico_sin_ia_muestra_cifras_limpias():
    r = C.responder_cliente("¿Cuánto debería ahorrar?", _datos(_cliente(gasto=2700.0)))
    assert r["modo"] == "basico" and "$600" in r["respuesta"] and "$300" in r["respuesta"]
    assert "Dato principal" not in r["respuesta"] and "| estado" not in r["respuesta"]


def test_respuesta_ia_valida():
    datos = _datos(_cliente(gasto=2700.0))
    llm = FakeListChatModel(responses=["Lo ideal es ahorrar $600 al mes; hoy te quedan $300.\n- Programa tu ahorro."])
    r = C.responder_cliente("¿Cuánto debería ahorrar?", datos, llm)
    assert r["modo"] == "ia" and not r["cifras_no_verificadas"]


def test_promesa_de_aprobacion_se_corrige():
    datos = _datos(_cliente())
    llm = FakeListChatModel(responses=["¡Claro! Te aprobamos el crédito sin problema.",
                                       "Una mensualidad nueva de hasta $480 cabría en la regla del 36%."])
    r = C.responder_cliente("¿Me aprueban un crédito?", datos, llm)
    assert r["modo"] == "ia" and r["reintentos"] == 1 and "aprobación" in r["problemas"][0]
    assert r["requiere_ejecutivo"]


def test_cifras_inventadas_dos_veces_van_a_respaldo():
    llm = FakeListChatModel(responses=["Ahorra $12,345 al mes.", "Ahorra $54,321 al mes."])
    r = C.responder_cliente("¿Cuánto debería ahorrar?", _datos(_cliente(gasto=2700.0)), llm)
    assert r["modo"] == "respaldo" and "$600" in r["respuesta"]


def test_mencionar_a_otro_cliente_es_un_problema():
    contexto = C.consulta_resumen(_datos(_cliente()))
    problemas = C.problemas_respuesta("Igual que el cliente U00777, vas bien.", contexto, "¿cómo voy?", "U00001")
    assert "mencionó a otro cliente" in problemas


@pytest.mark.parametrize("cambios, esperado", [
    ({}, "hasta $480"),                                      # 36% de 3,000 = 1,080 - 600 ya pagados
    ({"gasto": 2900.0}, "hasta $100"),                       # solo le sobran 100 al mes
    ({"gasto": 3300.0}, "no te conviene una deuda nueva"),   # gasta mas de lo que gana
    ({"pago_deuda": 1200.0}, "ya pasan del límite sano"),
])
def test_capacidad_de_pago_no_rebasa_lo_que_sobra(cambios, esperado):
    assert esperado in C.consulta_capacidad(_datos(_cliente(**cambios)))


def test_dificultad_pide_ejecutivo():
    r = C.responder_cliente("No puedo pagar mi mensualidad este mes", _datos(_cliente()))
    assert r["requiere_ejecutivo"] and "reestructurar" in r["respuesta"]


def test_todas_las_consultas_con_datos_reales(clientes):
    """Ninguna consulta truena con clientes reales (con y sin prestamo)."""
    for uid in [clientes[clientes["tiene_prestamo"] == 1]["user_id"].iloc[0],
                clientes[clientes["tiene_prestamo"] == 0]["user_id"].iloc[0]]:
        datos = _datos(C.perfil_cliente(clientes, uid), clientes)
        for tema in C.CONSULTAS:
            assert C.ejecutar_temas(datos, [tema]).strip()


# --- Evaluacion (S19) -------------------------------------------------------------
def test_evaluacion_en_modo_basico_cumple_la_meta(clientes):
    """Sin IA las respuestas son los datos exactos: enrutamiento, veracidad y
    exito deben ser 100%. Con IA, el mismo banco mide cuanto se pierde al redactar."""
    import evaluar_asesor as E

    tabla, resumen = E.evaluar(clientes, FECHA, llm=None)
    assert resumen["preguntas"] == 10 * len(C.clientes_ejemplo(clientes))
    fallas = tabla[~(tabla["enrutamiento_ok"] & tabla["exito_final"] & tabla["veraz_final"])]
    assert fallas.empty, fallas[["cliente", "pregunta", "respuesta"]].to_string()
    assert resumen["cumple_meta"]


def test_sla_del_asesor_se_lee_de_su_evaluacion(tmp_path):
    import json

    import observabilidad as O

    (tmp_path / "evaluacion_asesor.json").write_text(json.dumps({"exito_final_pct": 90.0}))
    entradas = O.cargar_entradas_locales(str(tmp_path))
    config = O.cargar_config()
    metricas = O.calcular_metricas({**entradas, "corridas": pd.DataFrame()}, config, pd.Timestamp.now(tz="America/Mexico_City"))
    assert metricas["evaluacion_asesor_exito_final_pct"] == 90.0


def test_preguntas_derivadas_no_cuentan_para_el_sla():
    import observabilidad as O

    ahora = pd.Timestamp("2026-11-16 12:00", tz="America/Mexico_City")
    log = pd.DataFrame({
        "timestamp": [ahora.isoformat()] * 4, "modo": ["ia", "ia", "derivado", "basico"],
        "canal": ["cliente", None, "cliente", "cliente"], "segundos": [10, 12, 0, 1], "reintentos": [0, 0, 0, 0]})
    m = O.metricas_asistente(log, ahora, 7)
    assert m["preguntas"] == 3 and m["derivadas"] == 1 and m["pct_ia"] == pytest.approx(66.7)
    assert m["por_canal"] == {"cliente": 2, "ejecutivo": 1}


# --- Mejoras tras la 1a evaluacion con Llama 3.2 (exito 55%) -------------------------
def test_la_cifra_principal_la_pone_el_sistema():
    """Aunque la IA solo de consejos generales (el fallo de la 1a evaluacion),
    la respuesta trae la cifra que se pregunto."""
    llm = FakeListChatModel(responses=["- Revisa tus gastos cada semana.\n- Aparta algo el día de pago."])
    r = C.responder_cliente("¿Cuánto debería ahorrar?", _datos(_cliente(gasto=2700.0)), llm)
    assert r["modo"] == "ia" and r["respuesta"].startswith("La meta es ahorrar $600 al mes")
    assert r["pasos"].startswith("- Revisa") and "Dato principal" not in r["respuesta"]


def test_la_cifra_se_muestra_antes_de_esperar_a_la_ia():
    vistos = []

    class LLMLento(FakeListChatModel):
        def _call(self, *args, **kwargs):
            assert vistos, "la interfaz debe recibir la cifra ANTES de que la IA responda"
            return super()._call(*args, **kwargs)

    llm = LLMLento(responses=["- Programa tu ahorro."])
    r = C.responder_cliente("¿Cuánto debería ahorrar?", _datos(_cliente(gasto=2700.0)), llm, al_tener_principal=vistos.append)
    assert vistos == [r["principal"]] and "$600" in vistos[0]


def test_derivados_no_llaman_al_callback():
    vistos = []
    C.responder_cliente("Me robaron la tarjeta", _datos(_cliente()), FakeListChatModel(responses=["x"]),
                        al_tener_principal=vistos.append)
    assert vistos == []


def test_siempre_hay_tres_clientes_de_ejemplo(clientes):
    """Si un perfil "de libro" no existe en los datos, se usa el mas parecido."""
    sin_deudas_pesadas = clientes.assign(pago_mensual_deuda_usd=clientes["pago_mensual_deuda_usd"].clip(upper=50))
    ejemplos = C.clientes_ejemplo(sin_deudas_pesadas)
    assert len(ejemplos) == 3 and len(set(ejemplos.values())) == 3


def test_evaluacion_con_ia_que_omite_cifras_cumple_la_meta(clientes):
    import evaluar_asesor as E

    llm = FakeListChatModel(responses=["- Revisa tus gastos cada semana.\n- Habla con tu banco si lo necesitas."])
    _, resumen = E.evaluar(clientes, FECHA, llm)
    assert resumen["preguntas"] == 30
    assert resumen["exito_final_pct"] >= 84 and resumen["veracidad_final_pct"] == 100


def test_configuracion_rapida_del_modelo():
    llm = C.crear_llm(url="http://127.0.0.1:9")
    assert llm.keep_alive == C.KEEP_ALIVE and llm.num_predict == C.MAX_TOKENS_PASOS and llm.temperature == 0


def test_calentar_sin_ollama_no_truena():
    assert C.calentar_modelo(url="http://127.0.0.1:9", timeout=1) is False
