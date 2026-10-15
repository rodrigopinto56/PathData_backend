"""
dashboard/ejecutivo.py

Semana 13: Dashboard ejecutivo v1.

Para quien NO es programador (p.ej. un ejecutivo de negocio): responde
en lenguaje de negocio cuatro preguntas, en este orden:
  1) ¿Como esta la cartera?          -> diagnostico en una frase + 4 KPIs clave
  2) ¿Como estan nuestros clientes?  -> semaforo verde / amarillo / rojo
  3) ¿Cuanto dinero esta en juego y donde esta el riesgo?
  4) ¿Que hacemos?                   -> acciones sugeridas + clientes prioritarios

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
except ModuleNotFoundError:  # ejecucion local fuera del contenedor
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import reporte_ejecutivo as rep

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
- **Fuente:** datos de demostración (dataset público sintético), actualizados cada día por el proceso automático de datos.
        """
    )
