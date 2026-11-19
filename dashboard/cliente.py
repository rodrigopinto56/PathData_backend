"""
dashboard/cliente.py

Semana 19: portal del CLIENTE con asesor financiero virtual (puerto 8503).

Para el cliente que llega al banco (o a la app) y necesita orientacion
financiera pero no tiene un ejecutivo disponible:

  1) Tu salud financiera   -> puntaje 0-100 y 4 pilares con semaforo
  2) Tu plan de accion     -> pasos concretos, con montos, en orden
  3) ¿Y si recorto gastos? -> simulador + presupuesto 50/30/20
  4) Personas como tu      -> medianas de grupos grandes (privacidad)
  5) Preguntale a tu asesor-> chat con IA local (Ollama), con reglas de
                              seguridad: fraude -> persona; otros clientes -> no

Toda la logica vive en scripts/asesor_cliente.py; aqui solo se dibuja.
El acceso con numero de cliente es de DEMOSTRACION: en produccion seria el
inicio de sesion de la banca en linea.
"""

from __future__ import annotations

import html
import os
import sys
import time

import pandas as pd
import streamlit as st

try:  # en el contenedor todo esta en /app; en local, los modulos viven en scripts/
    import asesor_cliente as ase
    import asistente_financiero as asis
    import observabilidad as obs
    import reporte_ejecutivo as rep
except ModuleNotFoundError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import asesor_cliente as ase
    import asistente_financiero as asis
    import observabilidad as obs
    import reporte_ejecutivo as rep

from data_loader import cargar_clientes_particion, cargar_gold_kpis

# --- Tokens (tema claro, mismos que el dashboard ejecutivo) ---
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
LINEA = "#e4e3df"
ACENTO = "#072146"
SERIE = "#2a78d6"
FUENTE_TITULOS = "'Source Serif 4', Georgia, 'Times New Roman', serif"
FUENTE_TEXTO = "'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
URL_FUENTES = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap"
)
# Semaforo: color + icono (nunca solo color)
ESTILO_ESTADO = {
    "bien": ("#1f8a4c", "✓"), "atencion": ("#c27a00", "!"), "urgente": ("#d03b3b", "✕"), "sin_datos": ("#8a8984", "?"),
}
COLOR_NIVEL = {"Sólida": "#1f8a4c", "En construcción": "#c27a00", "Frágil": "#d03b3b", "Sin datos": "#8a8984"}

st.set_page_config(page_title="Mi asesor financiero", layout="centered")
st.markdown(
    f"""
    <style>
      @import url('{URL_FUENTES}');
      .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
      .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stMetricLabel"] {{ font-family: {FUENTE_TEXTO}; }}
      .stApp h1, .stApp h2, .stApp h3 {{ font-family: {FUENTE_TITULOS}; color: {ACENTO}; letter-spacing: -0.01em; }}
      .stApp h1 {{ font-weight: 700; font-size: 2.2rem; }}
      .stApp [data-testid="stMetricValue"] {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
      .block-container {{ padding-top: 2rem; max-width: 860px; }}
      .muted {{ color:{TEXTO_2} !important; }}
      .score {{ background:#fff; border:1px solid {LINEA}; border-left:8px solid; border-radius:14px;
                padding:18px 22px; display:flex; align-items:center; gap:22px; flex-wrap:wrap; margin-bottom:12px; }}
      .score .num {{ font-family:{FUENTE_TEXTO}; font-size:3.2rem; font-weight:600; line-height:1; color:{TEXTO};
                     font-variant-numeric: tabular-nums; }}
      .score .num small {{ font-size:1.1rem; color:{TEXTO_2}; font-weight:500; }}
      .score .nivel {{ font-size:1.15rem; font-weight:600; }}
      .score p {{ margin:4px 0 0; color:{TEXTO}; }}
      .pilar {{ background:#fff; border:1px solid {LINEA}; border-radius:12px; padding:14px 16px; height:100%; margin-bottom:12px; }}
      .pilar-top {{ display:flex; justify-content:space-between; align-items:center; gap:8px; }}
      .pilar h4 {{ margin:0; font-family:{FUENTE_TEXTO}; font-size:0.78rem; font-weight:600; text-transform:uppercase;
                   letter-spacing:0.06em; color:{TEXTO_2}; }}
      .pilar .valor {{ font-size:1.7rem; font-weight:600; margin:6px 0 2px; color:{TEXTO}; font-variant-numeric: tabular-nums; }}
      .pilar p {{ margin:2px 0 0; font-size:0.88rem; color:{TEXTO}; }}
      .pilar .meta {{ font-size:0.78rem; color:{TEXTO_2}; margin-top:6px; }}
      .chip {{ display:inline-flex; align-items:center; gap:5px; border:1.5px solid; border-radius:999px;
               padding:1px 9px 1px 2px; font-size:0.75rem; font-weight:500; color:{TEXTO}; white-space:nowrap; }}
      .dot {{ display:inline-grid; place-items:center; width:17px; height:17px; border-radius:50%;
              color:#fff; font-size:0.7rem; font-weight:700; }}
      .accion {{ background:#fff; border:1px solid {LINEA}; border-radius:10px; padding:12px 16px; margin-bottom:8px; color:{TEXTO}; }}
      .accion b {{ color:{ACENTO}; }}
      .paso {{ display:inline-grid; place-items:center; width:24px; height:24px; border-radius:50%; background:{ACENTO};
               color:#fff; font-size:0.8rem; font-weight:700; margin-right:8px; }}
      .comp {{ margin:10px 0 14px; }}
      .comp-top {{ display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; color:{TEXTO}; font-size:0.92rem; }}
      .barra {{ height:8px; background:{LINEA}; border-radius:4px; margin-top:6px; overflow:hidden; }}
      .barra span {{ display:block; height:100%; background:{SERIE}; border-radius:4px; }}
      .aviso {{ background:#f4f7fb; border:1px solid #d6e0ee; border-radius:10px; padding:12px 16px; color:{TEXTO}; margin:8px 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Datos (solo la particion mas reciente: el cliente ve su foto de hoy)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def _fecha_reciente() -> str | None:
    kpis = cargar_gold_kpis()
    return None if kpis.empty else sorted(kpis["fecha"].unique())[-1]


@st.cache_data(ttl=300, show_spinner=False)
def _clientes(fecha: str) -> pd.DataFrame:
    return cargar_clientes_particion(fecha)


@st.cache_data(ttl=300, show_spinner=False)
def _ejemplos(fecha: str) -> dict:
    return ase.clientes_ejemplo(_clientes(fecha))


@st.cache_data(ttl=30, show_spinner=False)
def _estado_ollama() -> tuple[bool, str]:
    return asis.ollama_disponible()


@st.cache_resource
def _llm():
    return asis.crear_llm()


def _md(texto: str) -> str:
    """Streamlit interpreta $...$ como formula: se escapa el signo de pesos."""
    return texto.replace("$", "\\$")


def _e(texto) -> str:
    return html.escape(str(texto))


fecha = _fecha_reciente()
if fecha is None:
    st.title("Mi asesor financiero")
    st.info("Todavía no hay datos disponibles. Vuelve a intentarlo más tarde.")
    st.stop()
clientes = _clientes(fecha)
if clientes.empty:
    st.title("Mi asesor financiero")
    st.info("Todavía no hay datos de clientes para la fecha más reciente.")
    st.stop()

# ---------------------------------------------------------------------------
# Acceso (demostracion)
# ---------------------------------------------------------------------------
cliente_id = st.session_state.get("cliente_id")
if not cliente_id:
    st.title("Mi asesor financiero")
    st.markdown(
        '<p class="muted">Orientación personal sobre tu ahorro, tus deudas y tu presupuesto, a cualquier hora '
        "y sin esperar a un ejecutivo.</p>", unsafe_allow_html=True)
    with st.form("acceso"):
        numero = st.text_input("Número de cliente", placeholder="Por ejemplo: U00041")
        if st.form_submit_button("Entrar", type="primary"):
            if ase.perfil_cliente(clientes, numero) is None:
                st.error("No encontramos ese número de cliente. Revísalo e inténtalo de nuevo.")
            else:
                st.session_state["cliente_id"] = numero.strip().upper()
                st.rerun()
    ejemplos = _ejemplos(fecha)
    if ejemplos:
        st.markdown('<p class="muted">Para la demostración, entra como uno de estos clientes:</p>', unsafe_allow_html=True)
        columnas = st.columns(len(ejemplos))
        for columna, (nombre, uid) in zip(columnas, ejemplos.items()):
            if columna.button(f"{nombre}\n\n{uid}", use_container_width=True):
                st.session_state["cliente_id"] = uid
                st.rerun()
    st.caption("Demostración: en producción el acceso sería con el usuario y la contraseña de tu banca en línea.")
    st.stop()

perfil = ase.perfil_cliente(clientes, cliente_id)
if perfil is None:  # la particion cambio y el cliente ya no esta
    st.session_state.pop("cliente_id", None)
    st.rerun()

encabezado, salir = st.columns([5, 1])
encabezado.title("Hola, aquí está tu resumen")
encabezado.markdown(
    f'<p class="muted">Cliente {_e(perfil["user_id"])} · datos al {rep.fecha_legible(fecha)}</p>', unsafe_allow_html=True)
if salir.button("Salir"):
    st.session_state.pop("cliente_id", None)
    st.rerun()

# ---------------------------------------------------------------------------
# 1) Tu salud financiera
# ---------------------------------------------------------------------------
lista = ase.pilares(perfil)
puntaje, nivel = ase.puntaje_salud(lista)
color = COLOR_NIVEL[nivel]
frase = {
    "Sólida": "Tus finanzas están en buena forma. Lo importante ahora es mantener el rumbo.",
    "En construcción": "Vas bien en algunas cosas y hay otras que conviene atender pronto.",
    "Frágil": "Hay puntos que necesitan atención. Abajo tienes los pasos, en orden.",
    "Sin datos": "No tenemos datos suficientes para calcularla.",
}[nivel]
st.subheader("Tu salud financiera")
st.markdown(
    f'<div class="score" style="border-left-color:{color}"><div class="num">{puntaje if puntaje is not None else "–"}<small> / 100</small></div>'
    f'<div><div class="nivel" style="color:{color}">{_e(nivel)}</div><p>{_e(frase)}</p></div></div>',
    unsafe_allow_html=True,
)
for inicio in range(0, len(lista), 2):
    columnas = st.columns(2)
    for columna, pilar in zip(columnas, lista[inicio:inicio + 2]):
        c, icono = ESTILO_ESTADO[pilar.estado]
        columna.markdown(
            f'<div class="pilar"><div class="pilar-top"><h4>{_e(pilar.titulo)}</h4>'
            f'<span class="chip" style="border-color:{c}"><span class="dot" style="background:{c}">{icono}</span>{ase.ETIQUETA_ESTADO[pilar.estado]}</span></div>'
            f'<div class="valor">{_e(pilar.valor)}</div><p>{_e(pilar.explicacion)}</p>'
            f'<div class="meta">{_e(pilar.meta)}</div></div>',
            unsafe_allow_html=True,
        )
with st.expander("¿Cómo se calcula?"):
    st.markdown(_md(
        "Revisamos 4 pilares con reglas sencillas que usan los bancos y los educadores financieros:\n"
        "- **Ahorro mensual:** lo que te queda de tu ingreso después de tus gastos. Meta: 20% (regla 50/30/20).\n"
        "- **Fondo de emergencia:** cuántos meses de gastos cubren tus ahorros. Meta: al menos 3 meses.\n"
        "- **Deudas:** qué parte de tu ingreso se va en mensualidades. Lo sano es máximo 36%.\n"
        "- **Historial crediticio:** tu credit score. Con 670 o más consigues mejores condiciones.\n\n"
        "Cada pilar vale 25 puntos: 25 si está bien, 15 si requiere atención y 5 si es urgente."
    ))

# ---------------------------------------------------------------------------
# 2) Tu plan de accion
# ---------------------------------------------------------------------------
st.subheader("Tu plan de acción")
acciones = ase.plan_accion(perfil, lista)
for i, accion in enumerate(acciones, 1):
    st.markdown(
        f'<div class="accion"><span class="paso">{i}</span><b>{_e(accion["titulo"])}</b>'
        f'<div style="margin-top:6px">{_e(accion["detalle"])}</div></div>',
        unsafe_allow_html=True,
    )
if any(a["requiere_ejecutivo"] for a in acciones):
    st.markdown(
        '<div class="aviso"><b>Te conviene hablar con un ejecutivo.</b> Tu situación tiene opciones (como reestructurar '
        "un crédito) que solo puede revisar una persona del banco. Agenda una cita desde la app o en sucursal.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 3) ¿Y si recorto mis gastos? + presupuesto 50/30/20
# ---------------------------------------------------------------------------
st.subheader("¿Y si recorto mis gastos?")
recorte = st.slider("Recorte de tus gastos mensuales", 0, 30, 10, step=5, format="%d%%",
                    help="Mueve la barra para ver cuánto te quedaría al mes si gastaras menos")
simulacion = ase.simular_recorte(perfil, recorte)
if simulacion:
    s1, s2, s3 = st.columns(3)
    s1.metric("Te quedaría al mes", rep.fmt_usd(simulacion["flujo"]),
              delta=None if recorte == 0 else rep.fmt_usd(simulacion["flujo"] - perfil["flujo"]))
    s2.metric("Ahorro en 12 meses", rep.fmt_usd(simulacion["ahorro_12_meses"]))
    meses = simulacion["meses_para_fondo"]
    s3.metric("Fondo de emergencia (3 meses)",
              "Ya lo tienes" if meses == 0 else (f"en {meses} meses" if meses else "Aún no alcanza"))

presupuesto = ase.presupuesto_50_30_20(perfil)
if presupuesto:
    st.markdown("**Tu presupuesto ideal con la regla 50/30/20**")
    b1, b2, b3 = st.columns(3)
    b1.metric("Necesidades (50%)", rep.fmt_usd(presupuesto["necesidades"]), help="Renta, comida, transporte, servicios y mensualidades")
    b2.metric("Gustos (30%)", rep.fmt_usd(presupuesto["gustos"]), help="Salidas, compras y suscripciones")
    b3.metric("Ahorro (20%)", rep.fmt_usd(presupuesto["ahorro"]))
    diferencia = perfil["gasto"] - presupuesto["gasto_maximo"]
    st.caption(_md(
        f"Hoy gastas {rep.fmt_usd(perfil['gasto'])} al mes. "
        + ("Estás dentro del límite de necesidades + gustos." if diferencia <= 0
           else f"Eso es {rep.fmt_usd(diferencia)} más que el límite de necesidades + gustos "
                f"({rep.fmt_usd(presupuesto['gasto_maximo'])}).")))

# ---------------------------------------------------------------------------
# 4) Comparado con personas como tu
# ---------------------------------------------------------------------------
comparacion = ase.comparativo(perfil, clientes)
if comparacion:
    st.subheader("Comparado con personas como tú")
    st.markdown(
        f'<p class="muted">{_e(comparacion["grupo"].capitalize())} ({rep.fmt_int(comparacion["personas"])} personas). '
        "Usamos promedios de grupos grandes: nunca ves ni compartimos los datos de nadie.</p>",
        unsafe_allow_html=True,
    )
    for fila in comparacion["filas"]:
        lugar = fila["mejor_que_pct"]
        st.markdown(
            f'<div class="comp"><div class="comp-top"><span><b>{_e(fila["titulo"])}</b></span>'
            f'<span>Tú: <b>{_e(ase._fmt_indicador(fila["indicador"], fila["tu_valor"]))}</b> · '
            f'típico: {_e(ase._fmt_indicador(fila["indicador"], fila["mediana_grupo"]))}</span></div>'
            + (f'<div class="barra"><span style="width:{lugar}%"></span></div>'
               f'<div class="muted" style="font-size:0.8rem;margin-top:3px">Estás mejor que el {lugar}% del grupo</div>'
               if lugar is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# 5) Preguntale a tu asesor
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Pregúntale a tu asesor")
ia_disponible, _ = _estado_ollama()
st.caption(
    "Asesor con inteligencia artificial que corre en los servidores del banco: tus datos no salen a internet. "
    "Responde solo con tu información y con reglas de educación financiera; no aprueba créditos ni recomienda "
    "productos de inversión específicos."
)
if not ia_disponible:
    st.info("El asesor con IA no está disponible en este momento. Mientras tanto te respondo con tus datos exactos.")

datos_cliente = ase.DatosCliente(fecha=fecha, perfil=perfil, clientes=clientes)
historial = st.session_state.setdefault("chat_cliente", {}).setdefault(perfil["user_id"], [])

pregunta = None
columnas_ejemplo = st.columns(3)
for i, ejemplo in enumerate(ase.PREGUNTAS_EJEMPLO):
    if columnas_ejemplo[i % 3].button(ejemplo, key=f"ej_cliente_{i}", use_container_width=True):
        pregunta = ejemplo
pregunta = st.chat_input("Escribe tu pregunta (por ejemplo: ¿cómo pago mis deudas más rápido?)") or pregunta

if pregunta:
    with st.spinner("Revisando tus datos…"):
        inicio = time.time()
        resultado = ase.responder_cliente(pregunta, datos_cliente, _llm() if ia_disponible else None)
        historial.append({"pregunta": pregunta, **resultado})
        # Observabilidad (Vista 0). Privacidad: se registran el tema y el modo,
        # NO el texto de la pregunta ni el numero de cliente.
        obs.registrar_pregunta({
            "timestamp": pd.Timestamp.now(tz="America/Mexico_City").isoformat(), "fecha_datos": fecha,
            "canal": "cliente", "pregunta": "(cliente) " + ", ".join(resultado["temas"]),
            "modo": resultado["modo"], "segundos": round(time.time() - inicio, 1),
            "reintentos": resultado.get("reintentos", 0), "consultas": resultado["temas"],
            "cifras_no_verificadas": len(resultado.get("cifras_no_verificadas", [])),
            "modelo": asis.OLLAMA_MODEL if resultado["modo"] in ("ia", "respaldo") else None,
        })

for turno in historial[-6:]:
    with st.chat_message("user"):
        st.markdown(_md(turno["pregunta"]))
    with st.chat_message("assistant"):
        if turno["modo"] == "derivado":
            st.warning(_md(turno["respuesta"]))
        else:
            st.markdown(_md(turno["respuesta"]))
        if turno.get("requiere_ejecutivo") and turno["modo"] != "derivado":
            st.info("Para este tema te conviene hablar con un ejecutivo: agenda una cita desde la app o en sucursal.")
        if turno["modo"] in ("ia", "respaldo", "basico"):
            temas = ", ".join(ase.NOMBRE_TEMA.get(t, t) for t in turno["temas"])
            origen = "redactada con IA a partir de tus datos" if turno["modo"] == "ia" else "con tus datos exactos"
            st.caption(f"Revisé: {temas} · Respuesta {origen}.")
        if turno.get("cifras_no_verificadas"):
            st.caption("Algunas cifras no se pudieron confirmar con tus datos: tómalas como aproximadas.")
if historial and st.button("Borrar conversación"):
    historial.clear()
    st.rerun()

st.divider()
st.caption(
    "Información educativa basada en tus datos; no es una oferta de crédito ni una recomendación de inversión. "
    "Datos de demostración (dataset público sintético), actualizados cada día por el proceso automático de datos."
)
