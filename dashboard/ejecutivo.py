"""
dashboard/ejecutivo.py

Semana 13: Dashboard ejecutivo v1.

Para quien NO es programador (p.ej. un ejecutivo de negocio): responde
en lenguaje de negocio cuatro preguntas, en este orden:
  1) ¿Como esta la cartera?          -> diagnostico en una frase + 4 KPIs clave
  2) ¿Como estan nuestros clientes?  -> semaforo verde / amarillo / rojo
  3) ¿Cuanto dinero esta en juego y donde esta el riesgo?
  4) ¿Que hacemos?                   -> acciones sugeridas + clientes prioritarios
  5) ¿Y si las cosas se ponen dificiles? -> simulador de escenarios (Semana 16)

Es una app aparte del dashboard tecnico (app.py): mismo origen de datos
(gold-layer), distinta audiencia. Se sirve en el puerto 8502 con tema
claro (ver docker-compose.yml, servicio streamlit-ejecutivo).

La traduccion de KPIs a lenguaje de negocio vive en
scripts/reporte_ejecutivo.py, compartida con el reporte HTML que publica
el DAG cada dia -- asi el tablero y el reporte nunca dicen cosas distintas.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import reporte_ejecutivo as rep
    import simulador_montecarlo as sim
except ModuleNotFoundError:  # ejecucion local fuera del contenedor
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import reporte_ejecutivo as rep
    import simulador_montecarlo as sim

from data_loader import (
    cargar_clientes_particion,
    cargar_gold_kpis,
    cargar_gold_kpis_segmento,
    cargar_metadata_gold,
    cargar_tabla_particion,
)

# --- Tokens de tema claro ---
SURFACE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
LINEA = "#e4e3df"
ACENTO = "#072146"        # azul marino institucional para titulos
SERIE = "#2a78d6"         # una sola serie -> un solo color
# Pronostico del cliente demo: dos series (ingreso / gasto) -> categoricos
# slot 1 y slot 2 de la paleta, en ese orden fijo.
COLOR_SERIES = {"ingreso": "#2a78d6", "gasto": "#eb6834"}
RELLENO_BANDA = {"ingreso": "rgba(42,120,214,0.14)", "gasto": "rgba(235,104,52,0.14)"}

# --- Tipografia ---
# Titulos en serif editorial (como la lamina del proyecto) y texto/cifras en
# una sans corporativa con numeros tabulares (las cifras quedan alineadas).
# Si el navegador no puede bajar Google Fonts, cae a Georgia / Segoe UI.
FUENTE_TITULOS = "'Source Serif 4', Georgia, 'Times New Roman', serif"
FUENTE_TEXTO = "'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
URL_FUENTES = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600"
    "&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap"
)
UMBRAL_INDICE = 40        # meta del indice de riesgo

st.set_page_config(page_title="Salud financiera de la cartera", layout="wide")

st.markdown(
    f"""
    <style>
      @import url('{URL_FUENTES}');

      /* Texto general (sin tocar los iconos de Streamlit, que usan su propia fuente) */
      .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
      .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stMetricLabel"], .stApp [data-baseweb="select"] div {{
        font-family: {FUENTE_TEXTO};
      }}
      /* Titulos */
      .stApp h1, .stApp h2, .stApp h3 {{
        font-family: {FUENTE_TITULOS}; color: {ACENTO}; letter-spacing: -0.01em;
      }}
      .stApp h1 {{ font-weight: 700; font-size: 2.4rem; }}
      .stApp h3 {{ font-weight: 600; }}
      /* Cifras: sans, peso medio, numeros tabulares */
      .big, .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] * {{
        font-family: {FUENTE_TEXTO}; font-weight: 600;
        font-variant-numeric: tabular-nums lining-nums; letter-spacing: -0.02em; color: {TEXTO};
      }}
      .block-container {{ padding-top: 2rem; max-width: 1200px; }}
      .sub {{ color: {TEXTO_2}; margin-top: -0.6rem; }}
      .diag {{ background:#fff; border:1px solid {LINEA}; border-left:6px solid var(--c); border-radius:10px; padding:14px 18px; margin: 8px 0 4px; }}
      .diag p {{ margin:8px 0 0; font-size:1.05rem; color:{TEXTO}; }}
      .card {{ background:#fff; border:1px solid {LINEA}; border-radius:12px; padding:16px; height:100%; }}
      .card-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }}
      .card h4 {{ margin:10px 0 0; font-family:{FUENTE_TEXTO}; font-size:0.78rem; font-weight:600;
                  text-transform:uppercase; letter-spacing:0.06em; color:{TEXTO_2}; }}
      .big {{ font-size:2.1rem; margin:6px 0 2px; }}
      .card p {{ margin:4px 0 0; font-size:0.85rem; color:{TEXTO}; }}
      .muted {{ color:{TEXTO_2} !important; }}
      .chip {{ display:inline-flex; align-items:center; gap:5px; border:1.5px solid; border-radius:999px; padding:1px 9px 1px 2px; font-size:0.75rem; font-weight:500; color:{TEXTO}; white-space:nowrap; }}
      .dot {{ display:inline-grid; place-items:center; width:17px; height:17px; border-radius:50%; color:#fff; font-size:0.7rem; font-weight:700; }}
      .bar {{ display:flex; height:28px; border-radius:6px; overflow:hidden; gap:2px; }}
      .leg {{ display:flex; flex-wrap:wrap; gap:6px 28px; margin-top:10px; color:{TEXTO}; }}
      .sq {{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; }}
      .prog {{ height:6px; background:{LINEA}; border-radius:3px; margin-top:10px; overflow:hidden; }}
      .prog span {{ display:block; height:100%; border-radius:3px; }}
      .prog-txt {{ font-size:0.75rem !important; color:{TEXTO_2} !important; margin-top:4px !important;
                   font-variant-numeric: tabular-nums; }}
      .accion {{ background:#fff; border:1px solid {LINEA}; border-radius:10px; padding:12px 16px; margin-bottom:8px; color:{TEXTO}; }}
      .prio {{ display:inline-block; font-size:0.72rem; font-weight:700; border-radius:4px; padding:0 7px; margin-right:8px; border:1px solid {TEXTO_2}; color:{TEXTO_2}; }}
      .prio-alta {{ border-color:#d03b3b; color:#d03b3b; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def _kpis() -> pd.DataFrame:
    return cargar_gold_kpis()


@st.cache_data(ttl=60)
def _segmentos() -> pd.DataFrame:
    return cargar_gold_kpis_segmento()


@st.cache_data(ttl=60)
def _tabla(tabla: str, fecha: str) -> pd.DataFrame:
    return cargar_tabla_particion(tabla, fecha)


@st.cache_data(ttl=60)
def _metadata(carpeta: str, fecha: str) -> dict:
    return cargar_metadata_gold(carpeta, fecha)


@st.cache_data(ttl=60)
def _clientes(fecha: str) -> pd.DataFrame:
    return cargar_clientes_particion(fecha)


def _sin_latex(texto: str) -> str:
    """Streamlit interpreta $...$ como formula matematica. Dentro de bloques
    HTML el escape con barra se ve literal, asi que se usa la entidad &#36;."""
    return texto.replace("$", "&#36;")


def _chip(estado: str) -> str:
    color = rep.COLOR_ESTADO.get(estado, rep.COLOR_ESTADO[rep.SIN_DATOS])
    return (
        f'<span class="chip" style="border-color:{color}"><span class="dot" style="background:{color}">'
        f'{rep.ICONO_ESTADO.get(estado, "–")}</span>{estado}</span>'
    )


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
kpis = _kpis()
segmentos = _segmentos()

st.title("Salud financiera de la cartera")

if kpis.empty:
    st.info("Todavía no hay indicadores publicados. El equipo de datos debe ejecutar el proceso diario (transformacion_gold_dag).")
    st.stop()

fechas = sorted(kpis["fecha"].unique(), reverse=True)
col_sub, col_fecha = st.columns([3, 1])
with col_fecha:
    fecha = st.selectbox("Fecha de corte", fechas, index=0, format_func=rep.fecha_legible)

kpis_fecha = kpis[kpis["fecha"] == fecha]
seg_fecha = segmentos[segmentos["fecha"] == fecha] if not segmentos.empty else segmentos
resumen = rep.resumen_ejecutivo(kpis_fecha, seg_fecha)

if resumen is None:
    st.warning(
        "Los indicadores de esta fecha se calcularon con una versión anterior del proceso. "
        "Pide al equipo de datos que vuelva a ejecutar transformacion_gold_dag para esta fecha."
    )
    st.stop()

with col_sub:
    st.markdown(
        f'<p class="sub">Gemelo Digital Financiero · Corte al {rep.fecha_legible(fecha)} · '
        f'{rep.fmt_int(resumen["total_clientes"])} clientes</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 1) Diagnostico + KPIs clave
# ---------------------------------------------------------------------------
diag = resumen["diagnostico"]
st.markdown(
    f'<div class="diag" style="--c:{rep.COLOR_ESTADO[diag["nivel"]]}">{_chip(diag["nivel"])}<p>{diag["frase"]}</p></div>',
    unsafe_allow_html=True,
)

st.subheader("Indicadores clave")
columnas = st.columns(4)
for columna, t in zip(columnas, resumen["tarjetas"]):
    nota = f" · {t['nota']}" if t["nota"] else ""
    columna.markdown(
        f"""<div class="card">
              {_chip(t['estado'])}<h4>{t['titulo']}</h4>
              <div class="big">{t['valor']}</div>
              <p>{_sin_latex(t['explicacion'])}</p>
              {rep.barra_avance(t)}
              <p class="muted">{t['meta']}{nota}</p>
            </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 2) Semaforo de clientes + 3) Dinero en juego
# ---------------------------------------------------------------------------
st.subheader("¿Cómo están nuestros clientes?")
barra = "".join(
    f'<div style="width:{s["pct"] or 0}%;background:{s["color"]}" title="{s["etiqueta"]}: {rep.fmt_pct(s["pct"])}"></div>'
    for s in resumen["semaforo"]
)
leyenda = "".join(
    f'<span><span class="sq" style="background:{s["color"]}"></span>{s["etiqueta"]}: '
    f'<b>{rep.fmt_pct(s["pct"])}</b> ({rep.fmt_int(s["clientes"])} clientes)</span>'
    for s in resumen["semaforo"]
)
st.markdown(
    f'<div class="bar" role="img" aria-label="Clientes por nivel de riesgo">{barra}</div><div class="leg">{leyenda}</div>',
    unsafe_allow_html=True,
)
st.caption("Cada cliente recibe una calificación de riesgo de 0 a 100: verde menos de 40, amarillo de 40 a 69, rojo 70 o más.")

st.subheader("Dinero en juego")
d = resumen["dinero"]
c1, c2, c3 = st.columns(3)
c1.metric("Dinero prestado", rep.fmt_usd(d["exposicion"]), help="Suma de todos los préstamos vigentes.")
c1.caption(f"{rep.fmt_int(d['num_prestamos'])} préstamos activos")
c2.metric(
    "Pérdida esperada",
    rep.fmt_usd(d["perdida_esperada"]),
    help="Lo que estadísticamente se espera no recuperar: probabilidad de impago × monto × 45% (estándar de Basilea).",
)
c2.caption(f"Por cada \\$100 prestados se esperan perder \\${(d['perdida_por_cada_100'] or 0):.0f}")
c3.metric("Prestado a clientes de riesgo alto", rep.fmt_pct(d["pct_en_riesgo_alto"]), help="Porcentaje del dinero prestado que está en clientes con alta probabilidad de impago.")
c3.caption("del total prestado")

# ---------------------------------------------------------------------------
# ¿Donde esta el riesgo?
# ---------------------------------------------------------------------------
st.subheader("¿Dónde está el riesgo?")
dimensiones = [d for d in rep.NOMBRE_DIMENSION if not seg_fecha.empty and d in set(seg_fecha["dimension"])]
if dimensiones:
    dimension = st.radio(
        "Ver por", dimensiones, horizontal=True, format_func=lambda d: rep.NOMBRE_DIMENSION[d], label_visibility="collapsed"
    )
    tabla = rep.riesgo_por_segmento(seg_fecha, dimension)
    if "Índice de riesgo" in tabla.columns:
        grafica = tabla.sort_values("Índice de riesgo")
        fig = go.Figure(
            go.Bar(
                x=grafica["Índice de riesgo"],
                y=grafica["Segmento"],
                orientation="h",
                marker=dict(color=SERIE, cornerradius=4),
                text=grafica["Índice de riesgo"].round(0).astype(int),
                textposition="outside",
                textfont=dict(color=TEXTO),
                customdata=grafica[["Clientes", "% en rojo"]].to_numpy(),
                hovertemplate="<b>%{y}</b><br>Índice de riesgo: %{x:.0f}<br>Clientes: %{customdata[0]:,.0f}"
                "<br>En rojo: %{customdata[1]:.1f}%<extra></extra>",
            )
        )
        fig.add_vline(x=UMBRAL_INDICE, line_dash="dot", line_color=TEXTO_2)
        fig.add_annotation(
            x=UMBRAL_INDICE, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text=f"Meta: menos de {UMBRAL_INDICE}", font=dict(color=TEXTO_2, size=12),
        )
        fig.update_layout(
            plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font_color=TEXTO_2, showlegend=False,
            font_family=FUENTE_TEXTO, hoverlabel=dict(font_family=FUENTE_TEXTO),
            height=max(220, 52 * len(grafica) + 60), margin=dict(t=40, b=10, l=10, r=40),
            xaxis=dict(title="Índice de riesgo (0 = sano, 100 = riesgoso)", range=[0, 100], gridcolor=LINEA, zeroline=False),
            yaxis=dict(title=""),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla"):
        st.dataframe(
            tabla,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Clientes": st.column_config.NumberColumn(format="%d"),
                "Índice de riesgo": st.column_config.NumberColumn(format="%.0f"),
                "% en rojo": st.column_config.NumberColumn(format="%.1f%%"),
                "Dinero prestado": st.column_config.NumberColumn(format="$%.0f"),
                "Pérdida esperada": st.column_config.NumberColumn(format="$%.0f"),
                "Ahorro típico %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

# ---------------------------------------------------------------------------
# 4) ¿Que hacemos?
# ---------------------------------------------------------------------------
st.subheader("Acciones sugeridas")
st.caption("Generadas automáticamente a partir de los indicadores que están fuera de meta.")
for i, a in enumerate(resumen["acciones"], start=1):
    st.markdown(
        f'<div class="accion"><span class="prio prio-{a["prioridad"].lower()}">{a["prioridad"]}</span>'
        f'<b>{i}. {a["titulo"]}.</b> {_sin_latex(a["detalle"])}</div>',
        unsafe_allow_html=True,
    )

st.subheader("Clientes prioritarios")
clientes = _clientes(fecha)
if clientes.empty or "indice_riesgo_consolidado" not in clientes.columns:
    st.info("El detalle por cliente no está disponible para esta fecha.")
else:
    rojos = clientes[clientes["indice_riesgo_consolidado"] >= 70].sort_values("indice_riesgo_consolidado", ascending=False)
    lista = pd.DataFrame(
        {
            "Cliente": rojos["user_id"],
            "Índice de riesgo": rojos["indice_riesgo_consolidado"],
            "Prob. de impago %": rojos["riesgo_score"] * 100,
            "Principales factores": rojos.get("razones_riesgo", pd.Series("", index=rojos.index)),
            "Región": rojos["region"].map(rep.traducir),
            "Edad": rojos.get("age"),
            "Situación laboral": rojos.get("employment_status", pd.Series(dtype=str)).map(rep.traducir),
            "Ingreso mensual": rojos["monthly_income_usd"],
            "Préstamo": rojos["loan_amount_usd"],
            "Endeudamiento %": rojos["ratio_endeudamiento_mensual"] * 100,
            "Ahorro %": rojos["tasa_ahorro"] * 100,
        }
    )
    st.caption(
        f"{rep.fmt_int(len(lista))} clientes en rojo (índice de 70 o más), ordenados del más al menos riesgoso. "
        "Se muestran los primeros 15; descarga la lista completa abajo."
    )
    st.dataframe(
        lista.head(15),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Ingreso mensual": st.column_config.NumberColumn(format="$%.0f"),
            "Préstamo": st.column_config.NumberColumn(format="$%.0f"),
            "Endeudamiento %": st.column_config.NumberColumn(format="%.0f%%"),
            "Ahorro %": st.column_config.NumberColumn(format="%.0f%%"),
            "Prob. de impago %": st.column_config.NumberColumn(format="%.1f%%"),
            "Índice de riesgo": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
            "Principales factores": st.column_config.TextColumn(width="large"),
        },
    )
    st.download_button(
        "Descargar lista de clientes en rojo (CSV)",
        lista.to_csv(index=False).encode("utf-8-sig"),  # utf-8-sig: Excel respeta los acentos
        file_name=f"clientes_prioritarios_{fecha.replace('/', '-')}.csv",
        mime="text/csv",
    )

    # Que modelo esta detras de la probabilidad de impago
    modelo = _metadata("_gold_metadata", fecha).get("modelo", {})
    if modelo.get("origen") == "campeon_s15":
        st.caption(
            f"Probabilidad de impago estimada por el modelo de riesgo {modelo.get('version')} "
            f"({'scorecard' if modelo.get('modelo') == 'scorecard' else 'gradient boosting'}, "
            f"Gini {modelo.get('gini', 0):.2f} en datos de prueba). Los factores explican qué empuja "
            "el riesgo de cada cliente hacia arriba."
        )
    elif modelo:
        st.caption("Probabilidad de impago estimada con el modelo inicial (Semana 11); se actualizará al entrenar el modelo nuevo.")

# ---------------------------------------------------------------------------
# Cliente demo: proyeccion a 6 meses (Semana 15)
# ---------------------------------------------------------------------------
pronostico = _tabla("gold_pronostico_cliente_demo", fecha)
if not pronostico.empty:
    meta_pronostico = _metadata("_pronostico_metadata", fecha)
    st.subheader("Cliente demo: próximos 6 meses")
    st.caption(
        "Un cliente de demostración con 5 años de transacciones (no representa a la cartera). "
        "Línea continua: lo que pasó. Punteada: lo esperado, con su rango probable."
    )
    figura = go.Figure()
    meses_historia = sorted(pronostico.loc[pronostico["tipo"] == "historico", "mes"].unique())[-24:]
    todos_los_meses = meses_historia + sorted(pronostico.loc[pronostico["tipo"] == "pronostico", "mes"].unique())
    marcas = todos_los_meses[::6]  # una etiqueta cada 6 meses, en espanol
    for serie, etiqueta in [("ingreso", "Ingreso"), ("gasto", "Gasto")]:
        datos = pronostico[pronostico["serie"] == serie]
        historia = datos[(datos["tipo"] == "historico") & datos["mes"].isin(meses_historia)]
        futuro = datos[datos["tipo"] == "pronostico"]
        figura.add_trace(go.Scatter(
            x=list(futuro["mes"]) + list(futuro["mes"])[::-1],
            y=list(futuro["limite_superior"]) + list(futuro["limite_inferior"])[::-1],
            fill="toself", fillcolor=RELLENO_BANDA[serie], line=dict(width=0), mode="lines",
            hoverinfo="skip", showlegend=False,
        ))
        figura.add_trace(go.Scatter(
            x=historia["mes"], y=historia["valor"], name=etiqueta, mode="lines",
            line=dict(color=COLOR_SERIES[serie], width=2),
            hovertemplate=f"<b>{etiqueta}</b> %{{x}}<br>$%{{y:,.0f}}<extra></extra>",
        ))
        figura.add_trace(go.Scatter(
            x=[historia["mes"].iloc[-1], *futuro["mes"]], y=[historia["valor"].iloc[-1], *futuro["valor"]],
            name=f"{etiqueta} (pronóstico)", mode="lines+markers",
            line=dict(color=COLOR_SERIES[serie], width=2, dash="dot"), marker=dict(size=8),
            customdata=[[None, None], *futuro[["limite_inferior", "limite_superior"]].to_numpy().tolist()],
            hovertemplate=f"<b>{etiqueta} esperado</b> %{{x}}<br>$%{{y:,.0f}}"
                          "<br>Rango: $%{customdata[0]:,.0f} – $%{customdata[1]:,.0f}<extra></extra>",
        ))
    figura.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font_color=TEXTO_2, font_family=FUENTE_TEXTO,
        hoverlabel=dict(font_family=FUENTE_TEXTO), height=380, margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(gridcolor=LINEA, tickvals=marcas, ticktext=[rep.mes_legible(m) for m in marcas]),
        yaxis=dict(title="USD por mes", gridcolor=LINEA, tickformat="$,.0f"),
    )
    st.plotly_chart(figura, use_container_width=True)

    futuro_ahorro = pronostico[(pronostico["serie"] == "ahorro") & (pronostico["tipo"] == "pronostico")]
    meses_negativos = int((futuro_ahorro["valor"] < 0).sum())
    estab = meta_pronostico.get("estabilidad", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Ingreso esperado (promedio mensual)",
              rep.fmt_usd(pronostico.query("serie == 'ingreso' and tipo == 'pronostico'")["valor"].mean()))
    c2.metric("Gasto esperado (promedio mensual)",
              rep.fmt_usd(pronostico.query("serie == 'gasto' and tipo == 'pronostico'")["valor"].mean()))
    c3.metric("Meses con ahorro negativo", f"{meses_negativos} de {len(futuro_ahorro)}")
    if estab:
        st.caption(
            f"Variación mensual del ingreso: ±{estab.get('cv_ingreso_6m_pct', 0):.0f}% · del gasto: "
            f"±{estab.get('cv_gasto_6m_pct', 0):.0f}% (últimos 6 meses). Modelo de pronóstico elegido por "
            f"menor error histórico: {', '.join(f'{k}: {v}' for k, v in meta_pronostico.get('modelos_elegidos', {}).items())}."
        )

# ---------------------------------------------------------------------------
# 5) Simulador de escenarios (Semana 16, Monte Carlo)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def _config_simulador() -> dict:
    return sim.cargar_config()


@st.cache_data(ttl=600, show_spinner="Simulando 2,000 años posibles…")
def _simular_cartera_propia(fecha: str, multiplicador_pd: float, lgd: float, correlacion: str) -> dict:
    config = _config_simulador()["cartera"]
    pd_, ead = sim.preparar_cartera(_tabla("fact_posicion_financiera", fecha))
    escenario = {"multiplicador_pd": multiplicador_pd, "lgd": lgd,
                 "correlacion": correlacion if correlacion == "basilea" else float(correlacion)}
    resumen, perdidas = sim.simular_escenario_cartera(pd_, ead, escenario, config, config["simulaciones_interactivas"])
    resumen["perdidas"] = perdidas
    return resumen


def _layout(figura: go.Figure, alto: int, eje_x: dict, eje_y: dict) -> None:
    figura.update_layout(
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font_color=TEXTO_2, font_family=FUENTE_TEXTO,
        hoverlabel=dict(font_family=FUENTE_TEXTO), height=alto, margin=dict(t=56, b=10, l=10, r=10),
        showlegend=False, xaxis=dict(gridcolor=LINEA, zeroline=False, **eje_x), yaxis=dict(gridcolor=LINEA, **eje_y),
    )


def _linea_referencia(figura: go.Figure, x: float, texto: str, punteada: bool, fila: int) -> None:
    """Linea vertical con su etiqueta ARRIBA del area de la grafica (fila 0 o 1,
    para que dos etiquetas cercanas no se encimen ni tapen las barras)."""
    figura.add_vline(x=x, line_dash="dot" if punteada else "solid", line_color=TEXTO_2 if punteada else TEXTO, line_width=1.5)
    figura.add_annotation(x=x, y=1.0 + 0.09 * fila, yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                          text=f" {texto}", font=dict(color=TEXTO, size=12))


def grafica_perdidas(distribucion: pd.DataFrame, esperada: float, colchon: float) -> go.Figure:
    """Histograma: en cuantos de los anos simulados se perdio cada monto."""
    centro = (distribucion["desde"] + distribucion["hasta"]) / 2
    figura = go.Figure(go.Bar(
        x=centro, y=distribucion["pct_anos"], width=(distribucion["hasta"] - distribucion["desde"]) * 0.92,
        marker=dict(color=SERIE, cornerradius=3),
        customdata=distribucion[["desde", "hasta"]].to_numpy(),
        hovertemplate="Pérdida entre $%{customdata[0]:,.0f} y $%{customdata[1]:,.0f}"
                      "<br><b>%{y:.1f}% de los años</b><extra></extra>",
    ))
    _linea_referencia(figura, colchon, "Colchón actual", False, 0)
    _linea_referencia(figura, esperada, "Pérdida esperada", True, 1)
    _layout(figura, 340, dict(title="Pérdida en el año (USD)", tickformat="$~s"), dict(title="% de los años simulados"))
    return figura


def grafica_abanico(abanico: pd.DataFrame, saldo_inicial: float) -> go.Figure:
    """Abanico de saldo: rango del 90% (claro), del 50% (medio) y mediana."""
    meses = [0, *abanico["mes"]]
    def col(c):
        return [saldo_inicial, *abanico[c]]
    figura = go.Figure()
    for bajo, alto, alfa in [("p5", "p95", 0.12), ("p25", "p75", 0.26)]:
        figura.add_trace(go.Scatter(x=meses + meses[::-1], y=col(alto) + col(bajo)[::-1], fill="toself",
                                    fillcolor=f"rgba(42,120,214,{alfa})", line=dict(width=0), hoverinfo="skip"))
    figura.add_trace(go.Scatter(
        x=meses, y=col("p50"), mode="lines+markers", line=dict(color=SERIE, width=2), marker=dict(size=8),
        customdata=list(zip(col("p5"), col("p95"))),
        hovertemplate="Mes %{x}<br><b>Saldo típico: $%{y:,.0f}</b>"
                      "<br>9 de cada 10 futuros: $%{customdata[0]:,.0f} a $%{customdata[1]:,.0f}<extra></extra>",
    ))
    figura.add_hline(y=0, line_color=TEXTO, line_width=1.5)
    figura.add_annotation(x=0, y=0, xanchor="left", yanchor="bottom", showarrow=False,
                          text="Sin ahorro", font=dict(color=TEXTO, size=12))
    _layout(figura, 340, dict(title="Meses a partir de hoy", dtick=1), dict(title="Saldo ahorrado (USD)", tickformat="$,.0f"))
    return figura


def _texto_mes(mes) -> str:
    return "—" if mes is None or pd.isna(mes) else f"mes {mes:.0f}"


st.divider()
st.subheader("Simulador de escenarios: ¿y si las cosas se ponen difíciles?")
st.caption(
    "Simulación Monte Carlo: en lugar de un solo pronóstico, se simulan miles de futuros posibles "
    "y se cuenta qué tan seguido pasa cada cosa, sobre todo los casos malos."
)
simulacion_cartera = _tabla("gold_simulacion_cartera", fecha)
simulacion_cliente = _tabla("gold_simulacion_cliente_demo_resumen", fecha)
if simulacion_cartera.empty and simulacion_cliente.empty:
    st.info("La simulación de escenarios no está disponible para esta fecha. "
            "Pide al equipo de datos que vuelva a ejecutar transformacion_gold_dag.")
else:
    tab_cartera, tab_cliente = st.tabs(["Cartera de crédito", "Cliente demo"])

    with tab_cartera:
        if simulacion_cartera.empty:
            st.info("No hay préstamos con probabilidad de impago para simular en esta fecha.")
        else:
            escenarios = simulacion_cartera.set_index("escenario")
            base = escenarios.iloc[0]
            colchon = base["var_99_9"]  # reservas + capital que aguantan 1 ano de 1,000 en condiciones normales
            st.markdown(
                f"Cada escenario simula **{rep.fmt_int(base['simulaciones'])} años posibles** de la cartera "
                f"({rep.fmt_int(base['prestamos'])} préstamos, {_sin_latex(rep.fmt_usd(base['exposicion_total']))} prestados). "
                f"El **colchón actual** ({_sin_latex(rep.fmt_usd(colchon))}) es lo que el banco necesita hoy para "
                "aguantar un año de cada mil: reservas más capital.",
                unsafe_allow_html=True,
            )
            tabla_escenarios = pd.DataFrame({
                "Escenario": escenarios["nombre"],
                "Pérdida esperada": escenarios["perdida_esperada"].map(rep.fmt_usd),
                "Año muy malo (1 de 100)": escenarios["var_99"].map(rep.fmt_usd),
                "Peor caso (1 de 1,000)": escenarios["var_99_9"].map(rep.fmt_usd),
                "Prob. de rebasar el colchón": escenarios["prob_superar_colchon_pct"],
            })
            st.dataframe(
                tabla_escenarios, hide_index=True, use_container_width=True,
                column_config={"Prob. de rebasar el colchón": st.column_config.NumberColumn(format="%.1f%%")},
            )
            st.caption(" · ".join(f"**{e['nombre']}:** {e['descripcion']}" for _, e in escenarios.iterrows()))
            peor = escenarios.iloc[-1]
            st.markdown(
                f"**Lectura:** en el escenario **{peor['nombre'].lower()}** la pérdida esperada sube de "
                f"{_sin_latex(rep.fmt_usd(base['perdida_esperada']))} a {_sin_latex(rep.fmt_usd(peor['perdida_esperada']))} "
                f"y el colchón actual se rebasaría en **{peor['prob_superar_colchon_pct']:.0f}% de los años**.",
                unsafe_allow_html=True,
            )

            distribucion = _tabla("gold_simulacion_cartera_distribucion", fecha)
            if not distribucion.empty:
                elegido = st.radio("Distribución de pérdidas del escenario", list(escenarios.index), horizontal=True,
                                   format_func=lambda e: escenarios.loc[e, "nombre"], key="esc_cartera")
                st.plotly_chart(
                    grafica_perdidas(distribucion[distribucion["escenario"] == elegido],
                                     escenarios.loc[elegido, "perdida_esperada"], colchon),
                    use_container_width=True,
                )
                st.caption("Cada barra: en qué porcentaje de los años simulados se perdió ese monto. "
                           "Las barras a la derecha del colchón son los años en que no alcanzaría.")

            with st.form("escenario_cartera"):
                st.markdown("**Arma tu propio escenario**")
                f1, f2, f3 = st.columns(3)
                multiplicador = f1.slider("Impagos (veces los actuales)", 0.5, 4.0, 1.0, 0.1,
                                          help="1.0 = probabilidad de impago que estima hoy el modelo de riesgo.")
                lgd = f2.slider("Pérdida si un cliente no paga (%)", 20, 90, 45, 5,
                                help="Parte del préstamo que no se recupera. 45% es el estándar de Basilea.") / 100
                contagio = f3.select_slider(
                    "Contagio entre impagos", ["basilea", "0.10", "0.20", "0.30", "0.40"], value="basilea",
                    format_func=lambda v: "Normal (Basilea)" if v == "basilea" else f"{float(v):.0%}",
                    help="Qué tanto caen los clientes al mismo tiempo cuando la economía va mal.",
                )
                simular = st.form_submit_button("Simular", type="primary")
            if simular:
                propio = _simular_cartera_propia(fecha, multiplicador, lgd, contagio)
                # Se compara contra el escenario base simulado IGUAL (mismos 2,000 anos, misma
                # semilla): asi la diferencia es solo efecto de los sliders, no ruido de muestreo.
                base_vivo = _simular_cartera_propia(fecha, 1.0, 0.45, "basilea")
                prob_colchon = float((propio["perdidas"] > colchon).mean() * 100)

                def _vs_base(campo: str) -> str | None:
                    diferencia = propio[campo] - base_vivo[campo]
                    return None if abs(diferencia) < 1 else f"{rep.fmt_usd(diferencia)} vs. base"

                m1, m2, m3 = st.columns(3)
                m1.metric("Pérdida esperada", rep.fmt_usd(propio["perdida_esperada"]),
                          delta=_vs_base("perdida_esperada"), delta_color="inverse")
                m2.metric("Peor caso (1 de 1,000)", rep.fmt_usd(propio["var_99_9"]),
                          delta=_vs_base("var_99_9"), delta_color="inverse")
                m3.metric("Prob. de rebasar el colchón", rep.fmt_pct(prob_colchon))
                st.caption(f"{rep.fmt_int(propio['simulaciones'])} años simulados en vivo con las probabilidades de impago "
                           "de cada cliente; el resultado oficial del día (tabla de arriba) usa 10,000.")

    with tab_cliente:
        abanicos = _tabla("gold_simulacion_cliente_demo", fecha)
        if simulacion_cliente.empty or abanicos.empty:
            st.info("La fuente de transacciones del cliente demo no está disponible para esta fecha.")
        else:
            esc_cli = simulacion_cliente.set_index("escenario")
            config_cli = _config_simulador()["cliente_demo"]
            saldo_inicial = float(esc_cli["saldo_inicial"].iloc[0])
            st.markdown(
                f"Cada escenario arma **{rep.fmt_int(esc_cli['simulaciones'].iloc[0])} futuros de "
                f"{config_cli['horizonte_meses']} meses** con meses reales de su historial (los últimos "
                f"{config_cli['meses_historia']}) y sigue cuánto le queda ahorrado. Hoy se supone un ahorro de "
                f"{_sin_latex(rep.fmt_usd(saldo_inicial))} ({config_cli['saldo_inicial_meses_gasto']} meses de su gasto).",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame({
                    "Escenario": esc_cli["nombre"],
                    "Prob. de quedarse sin ahorro": esc_cli["prob_quedarse_sin_ahorro_pct"],
                    "Mes típico en que se agota": esc_cli["mes_mediano_agotamiento"].map(_texto_mes),
                    "Ahorro típico al final": esc_cli["saldo_final_mediana"].map(rep.fmt_usd),
                }),
                hide_index=True, use_container_width=True,
                column_config={"Prob. de quedarse sin ahorro": st.column_config.NumberColumn(format="%.1f%%")},
            )
            st.caption(" · ".join(f"**{e['nombre']}:** {e['descripcion']}" for _, e in esc_cli.iterrows()))
            elegido_cli = st.radio("Ver la evolución del ahorro en el escenario", list(esc_cli.index), horizontal=True,
                                   format_func=lambda e: esc_cli.loc[e, "nombre"], key="esc_cliente")
            st.plotly_chart(grafica_abanico(abanicos[abanicos["escenario"] == elegido_cli], saldo_inicial),
                            use_container_width=True)
            st.caption("Línea: el saldo típico (la mitad de los futuros queda por arriba y la mitad por abajo). "
                       "Franja oscura: la mitad de los futuros. Franja clara: 9 de cada 10.")

            if not pronostico.empty:
                serie_cliente = sim.serie_desde_pronostico(pronostico)
                with st.form("escenario_cliente"):
                    st.markdown("**Arma tu propio escenario**")
                    g1, g2, g3, g4 = st.columns(4)
                    cambio_ingreso = g1.slider("Cambio en su ingreso (%)", -50, 30, 0, 5) / 100
                    cambio_gasto = g2.slider("Cambio en su gasto (%)", -30, 50, 0, 5) / 100
                    sin_ingreso = g3.slider("Meses sin ingreso", 0, 6, 0)
                    ahorro_hoy = g4.number_input("Ahorro de hoy (USD)", 0, 1_000_000, int(round(saldo_inicial, -3)), 1000)
                    simular_cli = st.form_submit_button("Simular", type="primary")
                if simular_cli:
                    propio = sim.simular_cliente(
                        serie_cliente,
                        {"cambio_ingreso": cambio_ingreso, "cambio_gasto": cambio_gasto, "meses_sin_ingreso": sin_ingreso},
                        config_cli, saldo_inicial=float(ahorro_hoy),
                    )
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Prob. de quedarse sin ahorro", rep.fmt_pct(propio["prob_quedarse_sin_ahorro_pct"]))
                    k2.metric("Mes típico en que se agota", _texto_mes(propio["mes_mediano_agotamiento"]))
                    k3.metric("Ahorro típico al final", rep.fmt_usd(propio["saldo_final_mediana"]))
                    st.plotly_chart(grafica_abanico(propio["abanico"], propio["saldo_inicial"]), use_container_width=True)

            st.caption(
                "Supuestos: el dataset no trae el ahorro del cliente, por eso se parte de "
                f"{config_cli['saldo_inicial_meses_gasto']} meses de su gasto (ajustable arriba). Los meses se sortean "
                "de su historial real, así que no se repite la estacionalidad (p. ej. diciembre)."
            )

# ---------------------------------------------------------------------------
# Compartir + glosario
# ---------------------------------------------------------------------------
st.divider()
st.download_button(
    "Descargar reporte ejecutivo (1 página, imprimible)",
    rep.generar_reporte_html(resumen, seg_fecha).encode("utf-8"),
    file_name=f"reporte_ejecutivo_{fecha.replace('/', '-')}.html",
    mime="text/html",
    type="primary",
)

with st.expander("¿Cómo leer este tablero?"):
    st.markdown(
        """
- **Capacidad de ahorro:** qué parte de su ingreso le queda al cliente típico después de sus gastos. El *rango confiable* indica entre qué valores está el dato real con 95% de certeza.
- **Endeudamiento mensual:** qué parte del ingreso se va al pago de deudas, entre clientes con préstamo. Arriba de 36% el cliente está presionado.
- **Índice de riesgo:** calificación de 0 a 100 que combina endeudamiento (30%), probabilidad de impago estimada por un modelo (30%), capacidad de ahorro (20%) e historial crediticio (20%).
- **Pérdida esperada:** lo que estadísticamente se espera no recuperar; sirve para dimensionar reservas.
- **Estabilidad de ingresos y gastos:** cuánto cambian el ingreso y el gasto mes a mes (coeficiente de variación de los últimos 6 meses). Se mide en un *cliente demo* con 5 años de historial, porque la cartera no tiene historial mensual.
- **Avance hacia la meta:** la barra bajo cada indicador muestra qué tan cerca está de su meta (100% = cumplida) y cuántos puntos faltan o sobran.
- **Probabilidad de impago:** estimada por un modelo de riesgo entrenado con préstamos reales etiquetados y calibrado a una tasa de impago de cartera de 5%.
- **Simulación Monte Carlo:** se simulan miles de futuros posibles con las probabilidades de cada cliente y se cuenta qué tan seguido pasa cada resultado. El nombre viene del casino de Monte Carlo: la técnica usa el azar, como una ruleta, pero para medir riesgos.
- **Colchón actual:** reservas más capital que el banco necesita hoy para aguantar un año de cada mil (el peor caso 1 de 1,000 del escenario base). Es el criterio que usa la regulación bancaria (Basilea).
- **Peor caso (1 de 1,000):** pérdida que solo se supera en uno de cada mil años simulados. **Año muy malo (1 de 100):** igual, pero uno de cada cien.
- **Fuente:** datos de demostración (dataset público sintético), actualizados cada día por el proceso automático de datos.
        """
    )
