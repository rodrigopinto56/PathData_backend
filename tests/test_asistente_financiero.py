"""
tests/test_asistente_financiero.py

Semana 17: asistente financiero. No necesita Ollama: el LLM se reemplaza
por un modelo falso de LangChain y, para probar la integracion real con
ChatOllama, por un servidor HTTP local que imita la API de Ollama.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(os.path.dirname(__file__))

import asistente_financiero as A  # noqa: E402
import datos_sinteticos as ds  # noqa: E402
import prediccion_financiera as P  # noqa: E402
import simulador_montecarlo as M  # noqa: E402
from kpis_financieros import calcular_kpis_globales, calcular_kpis_por_segmento  # noqa: E402
from test_gold_kpis import FECHA, df_calificado, tablas  # noqa: E402,F401  (fixtures compartidas)


@pytest.fixture
def datos(tablas) -> A.DatosAsistente:
    """Las mismas tablas Gold que lee el dashboard, con datos sinteticos."""
    clientes = (
        tablas["fact_posicion_financiera"]
        .merge(tablas["dim_region"], on="region_id", how="left")
        .assign(razones_riesgo="Historial crediticio débil; Ingreso bajo")
    )
    config = M.cargar_config()
    serie = P.serie_mensual(ds.generar_transacciones())
    return A.DatosAsistente(
        fecha=FECHA,
        kpis=calcular_kpis_globales(tablas, FECHA),
        segmentos=calcular_kpis_por_segmento(tablas, FECHA),
        clientes=clientes,
        simulacion_cartera=M.simular_cartera(clientes, {**config["cartera"], "simulaciones": 500})["resumen"],
        simulacion_cliente=M.simular_cliente_escenarios(serie, {**config["cliente_demo"], "simulaciones": 300})["resumen"],
        pronostico=P.pronosticar(serie)["pronostico"],
    )


# --- 1) Enrutador -------------------------------------------------------------
@pytest.mark.parametrize("pregunta, esperada", [
    ("¿Cómo está la cartera hoy?", "resumen"),
    ("¿Qué región tiene más riesgo?", "segmentos"),
    ("¿Por qué el cliente U00041 es riesgoso?", "cliente"),
    ("¿A qué clientes debemos llamar primero?", "prioritarios"),
    ("¿Qué pasaría si hay una crisis?", "escenarios"),
    ("¿Cuánto podríamos perder?", "escenarios"),
    ("¿El cliente demo aguanta perder su empleo?", "cliente_demo"),
    ("¿Cuál es la capacidad de ahorro?", "indicadores"),
    ("hola", "resumen"),  # sin tema claro: resumen general
])
def test_enrutador_elige_la_consulta(pregunta, esperada):
    assert esperada in [consulta for consulta, _ in A.enrutar(pregunta)]


def test_enrutador_no_mezcla_cliente_demo_con_la_cartera():
    rutas = [c for c, _ in A.enrutar("¿Qué pasa si el cliente demo pierde su empleo?")]
    assert rutas == ["cliente_demo"]  # ni escenarios de la cartera ni la tabla por situacion laboral


def test_enrutador_detecta_dimension_y_cliente():
    assert ("segmentos", "rango_edad") in A.enrutar("¿Los jóvenes son más riesgosos por edad?")
    assert ("cliente", "U00041") in A.enrutar("dame el detalle de u00041")


# --- 2) Consultas -------------------------------------------------------------
def test_resumen_trae_diagnostico_kpis_y_dinero(datos):
    texto = A.consulta_resumen(datos)
    for fragmento in ["Diagnóstico general", "Indicadores clave", "Semáforo", "Dinero prestado", "pérdida esperada"]:
        assert fragmento in texto


def test_consulta_cliente_existente_e_inexistente(datos):
    cliente = datos.clientes[datos.clientes["tiene_prestamo"] == 1].iloc[0]
    texto = A.consulta_cliente(datos, cliente["user_id"])
    assert cliente["user_id"] in texto and "probabilidad de impago" in texto and "factores" in texto
    assert "No existe el cliente U99999" in A.consulta_cliente(datos, "U99999")


def test_prioritarios_son_los_de_mayor_indice(datos):
    texto = A.consulta_prioritarios(datos, cuantos=3)
    peor = datos.clientes.sort_values("indice_riesgo_consolidado", ascending=False).iloc[0]["user_id"]
    assert peor in texto.splitlines()[1]


def test_segmentos_en_espanol_y_ordenados(datos):
    texto = A.consulta_segmentos(datos, "region")
    assert "Norte" in texto or "Sur" in texto
    indices = [float(l.split("índice ")[1].split(",")[0]) for l in texto.splitlines()[1:]]
    assert indices == sorted(indices, reverse=True)


def test_indicadores_con_meta_en_palabras(datos):
    texto = A.consulta_indicadores(datos, ["capacidad_ahorro_mediana_pct"])
    assert "meta: mínimo 20%" in texto and "avance hacia la meta" in texto
    assert "Meta: 20%" not in texto  # la meta no se repite dentro de la descripcion


def test_escenarios_y_cliente_demo(datos):
    escenarios = A.consulta_escenarios(datos)
    assert all(nombre in escenarios for nombre in ["Base", "Adverso", "Severo"]) and "Colchón actual" in escenarios
    demo = A.consulta_cliente_demo(datos)
    assert "no representa a la cartera" in demo and "Desempleo" in demo


def test_tablas_faltantes_no_rompen(datos):
    vacio = A.DatosAsistente(fecha=FECHA, kpis=datos.kpis)
    for pregunta in A.PREGUNTAS_EJEMPLO + ["cliente U00001"]:
        resultado = A.responder(pregunta, vacio)
        assert resultado["respuesta"]


# --- 3) Respuesta con LLM (falso) y modo basico ----------------------------------
def test_modo_basico_sin_llm_regresa_los_datos(datos):
    resultado = A.responder("¿Qué región tiene más riesgo?", datos, llm=None)
    assert resultado["modo"] == "basico" and "Riesgo por región" in resultado["respuesta"]
    assert resultado["consultas"] == ["riesgo por segmento"]


def _cifra_real(datos) -> str:
    contexto = A.ejecutar_rutas(datos, A.enrutar("¿Cómo está la cartera?"))
    return A.PATRON_NUMERO.findall(contexto.split("Clientes en la cartera: ")[1])[0]


def test_respuesta_correcta_pasa_a_la_primera(datos):
    llm = FakeListChatModel(responses=[f"La cartera tiene {_cifra_real(datos)} clientes."])
    resultado = A.responder("¿Cómo está la cartera?", datos, llm=llm)
    assert resultado["modo"] == "ia" and resultado["reintentos"] == 0 and resultado["problemas"] == []


def test_cifra_inventada_se_corrige_con_un_reintento(datos):
    llm = FakeListChatModel(responses=[
        "La cartera perdería $987,654,321.",                      # 1er intento: cifra inventada
        f"La cartera tiene {_cifra_real(datos)} clientes.",       # reintento: correcto
    ])
    resultado = A.responder("¿Cómo está la cartera?", datos, llm=llm)
    assert resultado["modo"] == "ia" and resultado["reintentos"] == 1
    assert "987654321" in resultado["problemas"][0] and resultado["cifras_no_verificadas"] == []


def test_si_falla_dos_veces_muestra_los_datos_exactos(datos):
    llm = FakeListChatModel(responses=["Se perderían $987,654,321.", "Se perderían $555,555,555."])
    resultado = A.responder("¿Cómo está la cartera?", datos, llm=llm)
    assert resultado["modo"] == "respaldo" and "Diagnóstico general" in resultado["respuesta"]
    assert "987654321" not in resultado["respuesta"]


def test_control_detecta_negarse_con_datos_y_copiar_formato(datos):
    contexto = A.ejecutar_rutas(datos, A.enrutar("¿Cuál es la pérdida esperada?"))
    assert A.problemas_respuesta("No tengo información sobre la pregunta.", contexto, "")
    assert A.problemas_respuesta("DATOS:\n- Clientes: 5%", contexto, "")
    # Si de verdad no hay datos (cliente inexistente), decir "no tengo informacion" es correcto
    sin_datos = A.consulta_cliente(datos, "U99999")
    assert A.problemas_respuesta("No tengo información de ese cliente.", sin_datos, "") == []


def test_pregunta_de_perdida_del_ano_consulta_la_simulacion():
    rutas = [c for c, _ in A.enrutar("¿Cuál es la estimación de pérdida este año?")]
    assert rutas[:2] == ["indicadores", "escenarios"]


def test_indicadores_sin_formulas(datos):
    texto = A.consulta_indicadores(datos, ["perdida_esperada_usd", "perdida_esperada_pct_exposicion"])
    assert "sum(" not in texto and "PD x EAD" not in texto
    assert "Pérdida esperada de la cartera" in texto
    assert "Saludable (cumple la meta)" in texto or "Requiere atención (no cumple la meta)" in texto


def test_si_el_llm_falla_cae_a_modo_basico(datos):
    class LLMCaido(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise ConnectionError("Ollama no responde")

    resultado = A.responder("¿Cómo está la cartera?", datos, llm=LLMCaido(responses=["x"]))
    assert resultado["modo"] == "basico" and "Diagnóstico general" in resultado["respuesta"]
    assert "Ollama no responde" in resultado["error"]


def test_pregunta_vacia_y_pregunta_enorme(datos):
    assert "Escribe una pregunta" in A.responder("   ", datos)["respuesta"]
    resultado = A.responder("¿cómo está la cartera? " * 200, datos)
    assert resultado["respuesta"]


def test_verificador_ignora_numeros_cortos_y_acepta_formatos():
    contexto = "Pérdida esperada: $2.3 millones (5.1% de lo prestado). Clientes: 32,424."
    assert A.cifras_no_respaldadas("En 3 de cada 10 casos se pierden $2.3 millones, 5.1%, 32,424 clientes.", contexto) == []
    assert A.cifras_no_respaldadas("Se perderían $4.7 millones.", contexto) == ["4.7"]


# --- Integracion real con ChatOllama (servidor que imita la API de Ollama) --------
class _OllamaFalso(BaseHTTPRequestHandler):
    recibido: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/api/tags":
            self._json({"models": [{"name": "llama3.2:3b"}]})

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _OllamaFalso.recibido = cuerpo
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for linea in [
            {"model": cuerpo["model"], "created_at": "2026-11-03T00:00:00Z",
             "message": {"role": "assistant", "content": "La cartera está saludable."}, "done": False},
            {"model": cuerpo["model"], "created_at": "2026-11-03T00:00:00Z",
             "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ]:
            self.wfile.write((json.dumps(linea) + "\n").encode())

    def _json(self, datos):
        cuerpo = json.dumps(datos).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


@pytest.fixture
def ollama_falso():
    servidor = HTTPServer(("127.0.0.1", 0), _OllamaFalso)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{servidor.server_address[1]}"
    servidor.shutdown()


def test_integracion_chat_ollama(datos, ollama_falso):
    assert A.ollama_disponible(ollama_falso, "llama3.2:3b")[0]
    assert not A.ollama_disponible(ollama_falso, "llama3.1:8b")[0]  # modelo no descargado
    assert not A.ollama_disponible("http://127.0.0.1:9", timeout=0.5)[0]  # servidor apagado

    llm = A.crear_llm("llama3.2:3b", ollama_falso)
    resultado = A.responder("¿Cómo está la cartera?", datos, llm=llm)
    assert resultado["modo"] == "ia" and resultado["respuesta"] == "La cartera está saludable."
    mensajes = _OllamaFalso.recibido["messages"]
    assert [m["role"] for m in mensajes] == ["system", "user", "assistant", "user"]  # reglas + ejemplo + pregunta
    assert "Copia las cifras exactamente" in mensajes[0]["content"]
    assert "Diagnóstico general" in mensajes[3]["content"]  # los datos viajan en el prompt
    assert _OllamaFalso.recibido["options"]["temperature"] == 0


# --- Evaluacion automatica (scripts/evaluar_asistente.py) -------------------------
def test_evaluacion_enrutamiento_y_hechos_del_banco(datos):
    import evaluar_asistente as E

    tabla, resumen = E.evaluar(datos, None)  # modo basico: mide enrutamiento y que los datos contengan el hecho
    assert resumen["preguntas"] >= 30 and resumen["enrutamiento_pct"] >= 95
    assert resumen["exito_final_pct"] >= 95 and resumen["veracidad_final_pct"] == 100


def test_evaluacion_distingue_respuestas_buenas_y_malas(datos):
    import evaluar_asistente as E

    caso = E.Caso("¿Cuál es la pérdida esperada?", "indicadores", lambda d: E._fmt(d, "perdida_esperada_usd"))
    correcta = f"Se estima no recuperar {caso.hechos(datos)[0]}."
    _, buena = E.evaluar(datos, FakeListChatModel(responses=[correcta]), [caso])
    _, mala = E.evaluar(datos, FakeListChatModel(responses=["No tengo información.", "No tengo información."]), [caso])
    assert buena["exito_ia_pct"] == 100 and buena["veracidad_ia_primer_intento_pct"] == 100
    assert mala["exito_ia_pct"] == 0 and mala["veracidad_final_pct"] == 100  # el respaldo muestra datos exactos
