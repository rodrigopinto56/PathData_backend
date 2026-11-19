"""
dashboard/tema_bbva.py

Semana 19 (frontend): identidad visual UNICA para los tres tableros
(tecnico 8501, ejecutivo 8502 y portal del cliente 8503).

Todo lo visual vive aqui: paleta de BBVA, tipografia, encabezado con el
logo, titulos de seccion, tarjetas y estilo de las graficas de Plotly.
Si mañana cambia un color, se cambia en este archivo y los tres tableros
quedan iguales.

Paleta (azules corporativos de BBVA + colores de apoyo):
  - Navy #072146     -> encabezado, titulos (es el fondo del logo)
  - Core #004481     -> color primario de la interfaz (botones, enlaces)
  - Medio #1973B8    -> serie principal de las graficas
  - Aqua #02A5A5 / naranja #F7893B / morado #8F7AE5 -> series 2, 3 y 4
  La paleta categorica se valido con un script de daltonismo (separacion
  entre colores vecinos para deuteranopia, protanopia y tritanopia).
  - Estados (bien / atencion / critico): verde, ambar y rojo con icono
    y texto, nunca solo color.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache

import streamlit as st

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
NAVY = "#072146"
CORE = "#004481"
MEDIO = "#1973B8"
CLARO = "#5BBEFF"
AQUA = "#2DCCCD"
AQUA_OSCURO = "#02A5A5"
NARANJA = "#F7893B"
MORADO = "#8F7AE5"

FONDO = "#F4F6F9"          # fondo de la pagina
SUPERFICIE = "#FFFFFF"     # tarjetas y graficas
SUPERFICIE_2 = "#EEF2F7"   # tablas cebra, bloques suaves
TEXTO = "#121212"
TEXTO_2 = "#4D5A6B"
TEXTO_3 = "#7A8696"
LINEA = "#DFE5EC"
EJE = "#B8C2CE"

# Estados (reservados: nunca se usan como "serie 4")
VERDE = "#2E8B57"
AMBAR = "#D99A00"
ROJO = "#DA3851"
GRIS = "#8A94A3"

# Series de datos
SERIE = MEDIO                                          # una sola serie
CATEGORICA = [MEDIO, AQUA_OSCURO, NARANJA, MORADO]     # orden fijo, validado
SECUENCIAL = ["#B9DCF7", "#5BBEFF", "#1973B8", "#004481", NAVY]  # claro -> oscuro

# Tipografia: titulos en serif editorial (como los titulares de BBVA) y
# texto/cifras en una sans corporativa con numeros tabulares.
FUENTE_TITULOS = "'Source Serif 4', Georgia, 'Times New Roman', serif"
FUENTE_TEXTO = "'IBM Plex Sans', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
URL_FUENTES = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700"
    "&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap"
)

CARPETA_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
RUTA_ICONO = os.path.join(CARPETA_ASSETS, "bbva_icono.png")


@lru_cache(maxsize=None)
def logo_base64(variante: str = "blanco") -> str:
    """Logo como data URI (blanco para fondos navy, navy para fondos claros)."""
    ruta = os.path.join(CARPETA_ASSETS, f"bbva_logo_{variante}.png")
    try:
        with open(ruta, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Pagina
# ---------------------------------------------------------------------------
def configurar_pagina(titulo: str, layout: str = "wide", ancho_max: int = 1240) -> None:
    """set_page_config + hoja de estilos comun. Debe ser lo PRIMERO de cada app."""
    st.set_page_config(
        page_title=f"{titulo} · BBVA",
        page_icon=RUTA_ICONO if os.path.exists(RUTA_ICONO) else None,
        layout=layout,
    )
    st.markdown(_css(ancho_max), unsafe_allow_html=True)


def _css(ancho_max: int) -> str:
    return f"""
    <style>
      @import url('{URL_FUENTES}');

      /* --- Base --- */
      .stApp {{ background:{FONDO}; }}
      .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea,
      .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
      .stApp [data-testid="stMetricLabel"], .stApp [data-baseweb="select"] div,
      .stApp [data-baseweb="tab"] {{ font-family:{FUENTE_TEXTO}; }}
      .stApp p, .stApp li {{ color:{TEXTO}; }}
      .block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:{ancho_max}px; }}
      header[data-testid="stHeader"] {{ background:transparent; }}
      [data-testid="stDecoration"] {{ display:none; }}

      /* --- Titulos --- */
      .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family:{FUENTE_TITULOS}; color:{NAVY}; letter-spacing:-0.01em; }}
      .stApp h1 {{ font-weight:700; font-size:2.1rem; }}
      .stApp h2 {{ font-weight:600; font-size:1.6rem; }}
      .stApp h3 {{ font-weight:600; font-size:1.25rem; margin-top:0.6rem; }}
      .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p {{ color:{TEXTO_2}; }}

      /* --- Encabezado con logo --- */
      .bbva-header {{
        position:relative; overflow:hidden; display:flex; align-items:center; gap:22px; flex-wrap:wrap;
        padding:22px 28px; margin:0 0 22px; border-radius:14px; color:#fff;
        background:
          radial-gradient(circle at 92% 115%, rgba(45,204,205,0.55) 0%, rgba(45,204,205,0) 42%),
          radial-gradient(circle at 78% -30%, rgba(91,190,255,0.45) 0%, rgba(91,190,255,0) 45%),
          linear-gradient(115deg, {NAVY} 0%, {NAVY} 55%, #0B3B74 100%);
      }}
      .bbva-header::after {{
        content:""; position:absolute; top:-40%; bottom:-40%; right:24%; width:2px;
        background:linear-gradient(180deg, rgba(91,190,255,0) 0%, rgba(140,215,255,0.75) 50%, rgba(45,204,205,0) 100%);
        transform:rotate(28deg); pointer-events:none;
      }}
      .bbva-header img {{ height:30px; width:auto; display:block; }}
      .bbva-sep {{ width:1px; align-self:stretch; background:rgba(255,255,255,0.28); }}
      .bbva-producto {{ display:flex; flex-direction:column; gap:2px; min-width:220px; flex:1; }}
      .bbva-producto .eyebrow {{ font-family:{FUENTE_TEXTO}; font-size:0.72rem; font-weight:600; letter-spacing:0.14em;
                                text-transform:uppercase; color:{CLARO}; margin:0; }}
      .bbva-producto .titulo {{ font-family:{FUENTE_TITULOS}; font-size:1.65rem; font-weight:600; line-height:1.15;
                               color:#fff; margin:0; }}
      .bbva-producto .sub {{ font-family:{FUENTE_TEXTO}; font-size:0.88rem; color:rgba(255,255,255,0.78); margin:2px 0 0; }}
      .bbva-meta {{ position:relative; z-index:1; text-align:right; font-family:{FUENTE_TEXTO}; font-size:0.82rem;
                   color:rgba(255,255,255,0.85); line-height:1.5; }}
      .bbva-meta b {{ color:#fff; font-weight:600; }}

      /* --- Titulo de seccion --- */
      .bbva-seccion {{ margin:30px 0 14px; padding-top:18px; border-top:1px solid {LINEA}; }}
      .bbva-seccion.primera {{ border-top:none; padding-top:0; margin-top:6px; }}
      .bbva-seccion .eyebrow {{ font-family:{FUENTE_TEXTO}; font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                               text-transform:uppercase; color:{CORE}; margin:0 0 4px; }}
      .bbva-seccion .h2 {{ font-family:{FUENTE_TITULOS}; font-size:1.55rem; font-weight:600; color:{NAVY};
                          line-height:1.2; margin:0; letter-spacing:-0.01em; }}
      .bbva-seccion .desc {{ font-size:0.95rem; color:{TEXTO_2}; margin:6px 0 0; max-width:820px; }}

      /* --- Tarjetas de metricas de Streamlit --- */
      .stApp [data-testid="stMetric"] {{ background:{SUPERFICIE}; border:1px solid {LINEA}; border-radius:12px;
                                         padding:14px 16px; box-shadow:0 1px 2px rgba(7,33,70,0.04); }}
      .stApp [data-testid="stMetricLabel"] p {{ color:{TEXTO_2}; font-size:0.8rem; font-weight:600; letter-spacing:0.02em; }}
      .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricValue"] * {{
        font-family:{FUENTE_TEXTO}; font-weight:600; color:{TEXTO}; font-variant-numeric:tabular-nums lining-nums;
        letter-spacing:-0.02em; }}

      /* --- Pestañas --- */
      .stApp [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINEA}; }}
      .stApp [data-baseweb="tab"] {{ padding:10px 16px; font-weight:600; color:{TEXTO_2}; }}
      .stApp [data-baseweb="tab"][aria-selected="true"] {{ color:{CORE}; }}
      .stApp [data-baseweb="tab-highlight"] {{ background:{CORE}; height:3px; }}

      /* --- Expanders, graficas y tablas como tarjetas --- */
      .stApp [data-testid="stExpander"] details {{ background:{SUPERFICIE}; border:1px solid {LINEA}; border-radius:10px; }}
      .stApp [data-testid="stPlotlyChart"] {{ background:{SUPERFICIE}; border:1px solid {LINEA}; border-radius:12px;
                                              padding:8px 6px 2px; }}
      .stApp [data-testid="stDataFrame"] {{ border:1px solid {LINEA}; border-radius:10px; overflow:hidden; }}
      .stApp hr {{ border-color:{LINEA}; }}

      /* --- Botones --- */
      .stApp .stButton button, .stApp .stDownloadButton button {{ border-radius:8px; font-weight:600; }}
      /* Texto blanco en botones primarios (la regla de color de parrafos no debe ganarle) */
      .stApp button[kind^="primary"], .stApp button[kind^="primary"] p,
      .stApp [data-testid^="baseButton-primary"], .stApp [data-testid^="baseButton-primary"] p,
      .stApp [data-testid^="stBaseButton-primary"], .stApp [data-testid^="stBaseButton-primary"] p {{ color:#FFFFFF !important; }}
      .stApp .stButton button[kind="secondary"] {{ border-color:#C5D3E3; color:{CORE}; background:{SUPERFICIE}; }}
      .stApp .stButton button[kind="secondary"]:hover {{ border-color:{CORE}; background:#F0F6FC; }}

      /* --- Componentes propios --- */
      .bbva-card {{ background:{SUPERFICIE}; border:1px solid {LINEA}; border-radius:12px; padding:16px;
                   box-shadow:0 1px 2px rgba(7,33,70,0.04); height:100%; }}
      .bbva-chip {{ display:inline-flex; align-items:center; gap:5px; border:1.5px solid; border-radius:999px;
                   padding:1px 10px 1px 2px; font-size:0.75rem; font-weight:600; color:{TEXTO}; white-space:nowrap;
                   background:{SUPERFICIE}; }}
      .bbva-dot {{ display:inline-grid; place-items:center; width:17px; height:17px; border-radius:50%; color:#fff;
                  font-size:0.7rem; font-weight:700; }}
      .bbva-pie {{ margin-top:36px; padding-top:16px; border-top:1px solid {LINEA}; display:flex; align-items:center;
                  gap:14px; flex-wrap:wrap; color:{TEXTO_3}; font-size:0.8rem; }}
      .bbva-pie img {{ height:14px; width:auto; opacity:0.85; }}
    </style>
    """


def encabezado(titulo: str, eyebrow: str = "Gemelo Digital Financiero", subtitulo: str = "",
               meta_html: str = "", destino=None) -> None:
    """Franja navy con el logo de BBVA, nombre del producto y datos a la derecha.
    `meta_html` es HTML de confianza armado por la app (fecha de corte, cliente...).
    `destino`: un st.empty() para dibujarlo arriba aunque se calcule despues."""
    (destino or st).markdown(html_encabezado(titulo, eyebrow, subtitulo, meta_html), unsafe_allow_html=True)


def html_encabezado(titulo: str, eyebrow: str, subtitulo: str = "", meta_html: str = "") -> str:
    logo = logo_base64("blanco")
    imagen = f'<img src="{logo}" alt="BBVA">' if logo else '<b style="font-size:1.6rem;letter-spacing:0.08em">BBVA</b>'
    sub = f'<p class="sub">{subtitulo}</p>' if subtitulo else ""
    meta = f'<div class="bbva-meta">{meta_html}</div>' if meta_html else ""
    return (
        f'<div class="bbva-header">{imagen}<div class="bbva-sep"></div>'
        f'<div class="bbva-producto"><p class="eyebrow">{eyebrow}</p><p class="titulo">{titulo}</p>{sub}</div>'
        f"{meta}</div>"
    )


def seccion(titulo: str, eyebrow: str = "", descripcion: str = "", primera: bool = False) -> None:
    """Titulo de seccion uniforme: etiqueta pequeña + titulo serif + descripcion opcional."""
    partes = [f'<p class="eyebrow">{eyebrow}</p>' if eyebrow else "", f'<p class="h2">{titulo}</p>',
              f'<p class="desc">{descripcion}</p>' if descripcion else ""]
    clase = "bbva-seccion primera" if primera else "bbva-seccion"
    st.markdown(f'<div class="{clase}">{"".join(partes)}</div>', unsafe_allow_html=True)


def chip(texto: str, color: str, icono: str) -> str:
    """Estado con icono + texto (nunca solo color)."""
    return (f'<span class="bbva-chip" style="border-color:{color}"><span class="bbva-dot" '
            f'style="background:{color}">{icono}</span>{texto}</span>')


def pie_de_pagina(texto: str) -> None:
    logo = logo_base64("navy")
    imagen = f'<img src="{logo}" alt="BBVA">' if logo else ""
    st.markdown(f'<div class="bbva-pie">{imagen}<span>{texto}</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Graficas (Plotly)
# ---------------------------------------------------------------------------
def estilo_plotly(fig, mostrar_leyenda: bool = True, alto: int | None = None):
    """Mismo fondo, tipografia, rejilla y orden de colores en todas las graficas."""
    fig.update_layout(
        plot_bgcolor=SUPERFICIE,
        paper_bgcolor=SUPERFICIE,
        font=dict(family=FUENTE_TEXTO, color=TEXTO_2, size=13),
        colorway=CATEGORICA,
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXTO_2)),
        showlegend=mostrar_leyenda,
        hoverlabel=dict(bgcolor=SUPERFICIE, bordercolor=LINEA, font=dict(family=FUENTE_TEXTO, color=TEXTO)),
        margin=dict(t=36 if mostrar_leyenda else 14, b=10, l=10, r=14),
    )
    if fig.layout.title.text:  # solo si la grafica tiene titulo (si no, Plotly muestra "undefined")
        fig.update_layout(title_font=dict(family=FUENTE_TITULOS, color=NAVY, size=16))
    if alto:
        fig.update_layout(height=alto)
    fig.update_xaxes(gridcolor=LINEA, linecolor=EJE, zerolinecolor=EJE, tickfont=dict(color=TEXTO_2),
                     title_font=dict(color=TEXTO_2))
    fig.update_yaxes(gridcolor=LINEA, linecolor=EJE, zerolinecolor=EJE, tickfont=dict(color=TEXTO_2),
                     title_font=dict(color=TEXTO_2))
    fig.for_each_annotation(lambda a: a.update(font=dict(color=TEXTO_2, family=FUENTE_TEXTO)))
    return fig
