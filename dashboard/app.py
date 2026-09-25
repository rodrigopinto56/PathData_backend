"""
dashboard/app.py

Semana 10: Dashboard de Calidad - Gemelo Digital Financiero.

Estructura del archivo (de arriba a abajo):
  1) Chequeo de configuracion       (Paso 2 - "hola mundo")
  2) Vista 1: Salud de la malla     (Paso 4)
  3) Vista 2: Calidad de datos      (Paso 5)
  4) Vista 3: Riesgo de credito     (Semana 11 - capa Gold)

Cada vista es una funcion aparte para que crezcan sin pisarse entre si.
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from data_loader import (
    cargar_eventos_pipeline,
    cargar_reportes_calidad,
    cargar_gold_metadata_modelo,
    cargar_gold_metricas_segmento,
    cargar_gold_clientes_riesgo,
)

# --- Paleta fija del proyecto (misma en toda vista, nunca improvisada) ---
# Tokens en modo oscuro, porque el tema por default de Streamlit es oscuro.
# Sacados de la misma paleta de referencia, columna "Dark".
SURFACE = "#1a1a19"        # fondo de la grafica (mismo que el fondo de Streamlit)
TEXTO_PRIMARIO = "#ffffff"
TEXTO_SECUNDARIO = "#c3c2b7"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"

# 'nivel' es un estado (INFO = todo bien, ALERTA = algo fallo), por eso usa
# la paleta de "status", no colores categoricos genericos.
COLOR_STATUS = {"INFO": "#0ca30c", "ALERTA": "#d03b3b", "OTRO": TEXTO_SECUNDARIO}
COLOR_SERIE_PRINCIPAL = "#3987e5"  # slot 1 categorico, paso oscuro

# 'fuente' si es identidad (dos datasets distintos, sin orden ni jerarquia
# entre ellos), por eso usa colores categoricos en orden fijo (slot 1, slot 2),
# nunca el mismo verde/rojo de 'nivel' - es una serie distinta, color distinto.
COLOR_FUENTE = {"loan_default_risk": "#3987e5", "personal_finance_ml": "#d95926"}
UMBRAL_CALIDAD_OK = 95.0  # % minimo antes de considerar una particion en riesgo

# 'rango_credit_score' SI tiene un orden (Bajo -> Excelente), por eso usa
# una rampa secuencial de un solo tono (mas oscuro = mejor score) en vez de
# colores categoricos sin relacion entre si -- ver gold_schema.yml.
ORDEN_BANDAS = ["Bajo", "Regular", "Bueno", "Muy bueno", "Excelente"]
COLOR_BANDAS = {
    "Bajo": "#86b6ef",
    "Regular": "#5598e7",
    "Bueno": "#2a78d6",
    "Muy bueno": "#1c5cab",
    "Excelente": "#104281",
}
COLOR_RIESGO = {"Alto": "#d03b3b", "Bajo": "#0ca30c"}  # mismo par que 'nivel': es un status

st.set_page_config(page_title="Gemelo Digital Financiero - Dashboard", layout="wide")

st.title("Dashboard de Calidad - Gemelo Digital Financiero")


# --- Cache de datos ---
# Sin esto, cada seccion que llama a cargar_eventos_pipeline() o
# cargar_reportes_calidad() vuelve a leer el archivo / a golpear MinIO,
# y ahora que el resumen ejecutivo tambien las necesita serian 2-3 lecturas
# por cada refresco de pantalla. @st.cache_data guarda el resultado en
# memoria por `ttl` segundos: dentro de esa ventana, todas las secciones
# comparten el mismo DataFrame ya leido.
@st.cache_data(ttl=30)
def _eventos_cacheados() -> pd.DataFrame:
    return cargar_eventos_pipeline()


@st.cache_data(ttl=30)
def _reportes_cacheados() -> pd.DataFrame:
    return cargar_reportes_calidad()


@st.cache_data(ttl=30)
def _gold_metadata_cacheada() -> pd.DataFrame:
    return cargar_gold_metadata_modelo()


@st.cache_data(ttl=30)
def _gold_segmento_cacheado() -> pd.DataFrame:
    return cargar_gold_metricas_segmento()


@st.cache_data(ttl=30)
def _gold_clientes_cacheados() -> pd.DataFrame:
    return cargar_gold_clientes_riesgo()


def _tema_oscuro(fig, mostrar_leyenda: bool = True):
    """Aplica el mismo fondo/tipografia que el resto de la app a una figura
    de Plotly. Sin esto, Plotly deja fondo blanco por default y se ve como
    una caja pegada encima de la interfaz oscura de Streamlit."""
    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font_color=TEXTO_SECUNDARIO,
        legend_title_text="",
        showlegend=mostrar_leyenda,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRIDLINE, linecolor=BASELINE, zerolinecolor=BASELINE)
    return fig


# ---------------------------------------------------------------------------
# 1) Chequeo de configuracion (Paso 2)
# ---------------------------------------------------------------------------
def seccion_chequeo_configuracion() -> None:
    with st.expander("Chequeo de configuracion", expanded=False):
        minio_endpoint = os.getenv("MINIO_ENDPOINT", "NO DEFINIDA")
        minio_bucket = os.getenv("SILVER_BUCKET", "NO DEFINIDA")

        col1, col2 = st.columns(2)
        col1.metric("MINIO_ENDPOINT", minio_endpoint)
        col2.metric("SILVER_BUCKET", minio_bucket)

        if minio_endpoint == "NO DEFINIDA" or minio_bucket == "NO DEFINIDA":
            st.warning(
                "Faltan variables de entorno. Revisa el bloque 'environment:' "
                "del servicio streamlit en docker-compose.yml."
            )


# ---------------------------------------------------------------------------
# 2) Vista 1: Salud de la malla (Paso 4)
# ---------------------------------------------------------------------------
def seccion_salud_malla() -> None:
    st.header("Vista 1 - Salud de la malla")
    st.caption(
        "De donde sale esto: cada tarea de tus DAGs (ingesta_bronze_dag y "
        "transformacion_silver_dag) escribe un evento en pipeline_log.jsonl "
        "cuando corre. Aqui se cuentan esos eventos."
    )

    eventos = _eventos_cacheados().copy()

    if eventos.empty:
        st.info(
            "Todavia no hay eventos registrados. Corre un backfill de "
            "transformacion_silver_dag para ver datos aqui."
        )
        return

    # --- Normalizacion de 'nivel' ---
    # Se vio en el dashboard un tercer color en la leyenda ademas de
    # INFO/ALERTA: significa que algun evento trae un valor de 'nivel'
    # que no es ninguno de los dos (por ejemplo, vacio o de otro DAG que
    # todavia no usa exactamente estos dos textos). En vez de dejar que
    # Plotly le invente un color random sin explicacion, lo agrupamos
    # explicitamente bajo "OTRO" y lo avisamos, para poder ir a revisar
    # el pipeline_log.jsonl con el detalle si hace falta.
    if "nivel" in eventos.columns:
        valores_conocidos = {"INFO", "ALERTA"}
        eventos["nivel"] = eventos["nivel"].fillna("OTRO")
        valores_encontrados = set(eventos["nivel"].unique())
        valores_inesperados = valores_encontrados - valores_conocidos
        if valores_inesperados:
            eventos.loc[eventos["nivel"].isin(valores_inesperados), "nivel"] = "OTRO"
            st.caption(
                f"Nota: se encontraron valores de 'nivel' distintos a INFO/ALERTA "
                f"({', '.join(sorted(valores_inesperados))}), agrupados aqui como 'OTRO'."
            )

    # --- KPIs (numeros grandes, primero, para responder rapido "vamos bien?") ---
    total_eventos = len(eventos)
    eventos_alerta = int((eventos["nivel"] == "ALERTA").sum()) if "nivel" in eventos else 0
    tasa_exito = round((1 - eventos_alerta / total_eventos) * 100, 1) if total_eventos else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos totales", f"{total_eventos:,}")
    col2.metric("Tasa de exito", f"{tasa_exito}%")
    col3.metric("Eventos en ALERTA", eventos_alerta)

    # --- Eventos por dia, coloreados por nivel (INFO vs ALERTA) ---
    # Por que barras apiladas y no una linea: lo que importa aqui no es la
    # tendencia fina, es "cuantos eventos, y que proporcion fueron alerta,
    # cada dia" - una pregunta de composicion, no de trayectoria.
    if "timestamp" in eventos.columns and "nivel" in eventos.columns:
        eventos_validos = eventos.dropna(subset=["timestamp"]).copy()
        eventos_validos["fecha"] = eventos_validos["timestamp"].dt.date

        conteo_diario = (
            eventos_validos.groupby(["fecha", "nivel"]).size().reset_index(name="conteo")
        )

        fig = px.bar(
            conteo_diario,
            x="fecha",
            y="conteo",
            color="nivel",
            category_orders={"nivel": ["INFO", "ALERTA", "OTRO"]},
            color_discrete_map=COLOR_STATUS,
            barmode="stack",
            labels={"fecha": "Fecha", "conteo": "Eventos", "nivel": "Nivel"},
        )
        st.plotly_chart(_tema_oscuro(fig), use_container_width=True)

    # --- Eventos por tarea (que tarea corre mas / cual genera mas alertas) ---
    if "task" in eventos.columns:
        conteo_por_tarea = (
            eventos.groupby("task").size().reset_index(name="conteo").sort_values("conteo")
        )
        fig_tareas = px.bar(
            conteo_por_tarea,
            x="conteo",
            y="task",
            orientation="h",
            text="conteo",
            labels={"conteo": "Eventos", "task": "Tarea"},
        )
        fig_tareas.update_traces(
            marker_color=COLOR_SERIE_PRINCIPAL,
            textposition="outside",
            textfont_color=TEXTO_PRIMARIO,
        )
        st.plotly_chart(_tema_oscuro(fig_tareas, mostrar_leyenda=False), use_container_width=True)

    # --- Tabla de las alertas mas recientes (para saber que revisar primero) ---
    if eventos_alerta:
        st.subheader("Ultimas alertas")
        columnas_disponibles = [
            c for c in ["timestamp", "task", "fuente", "particion"] if c in eventos.columns
        ]
        tabla_alertas = (
            eventos[eventos["nivel"] == "ALERTA"][columnas_disponibles]
            .sort_values("timestamp", ascending=False)
            .head(10)
        )
        st.dataframe(tabla_alertas, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 3) Vista 2: Calidad de datos (Paso 5)
# ---------------------------------------------------------------------------
def seccion_calidad_datos() -> None:
    st.header("Vista 2 - Calidad de datos")
    st.caption(
        "De donde sale esto: cada particion (fuente + fecha) que pasa por "
        "transformacion_silver_dag genera un _calidad_reporte.json en MinIO, "
        "con el resultado de las expectativas de Great Expectations definidas "
        "en config/silver_schema.yml."
    )

    reportes = _reportes_cacheados()

    if reportes.empty:
        st.info(
            "Todavia no hay reportes de calidad en MinIO. Corre un backfill "
            "de transformacion_silver_dag para ver datos aqui."
        )
        return

    # --- KPIs ---
    calidad_promedio = round(reportes["porcentaje_calidad"].mean(), 1)
    particiones_con_fallas = int((reportes["expectativas_fallidas"] > 0).sum())
    total_particiones = len(reportes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Calidad promedio", f"{calidad_promedio}%")
    col2.metric("Particiones evaluadas", total_particiones)
    col3.metric("Particiones con fallas", particiones_con_fallas)

    # --- % de calidad por fecha, una barra por fuente ---
    # Barras agrupadas (no apiladas): aqui SI importa comparar el valor
    # exacto de cada fuente lado a lado, no su suma - sumar dos porcentajes
    # no significa nada.
    fig = px.bar(
        reportes.sort_values("fecha"),
        x="fecha",
        y="porcentaje_calidad",
        color="fuente",
        barmode="group",
        color_discrete_map=COLOR_FUENTE,
        labels={
            "fecha": "Particion (fecha)",
            "porcentaje_calidad": "% expectativas cumplidas",
            "fuente": "Fuente",
        },
    )
    fig.update_yaxes(range=[0, 105])
    fig.add_hline(
        y=UMBRAL_CALIDAD_OK,
        line_dash="dot",
        line_color=TEXTO_SECUNDARIO,
        annotation_text=f"Umbral {UMBRAL_CALIDAD_OK}%",
        annotation_font_color=TEXTO_SECUNDARIO,
    )
    st.plotly_chart(_tema_oscuro(fig), use_container_width=True)

    # --- Detalle de particiones con fallas (para saber que revisar primero) ---
    st.subheader("Particiones con expectativas fallidas")
    con_fallas = reportes[reportes["expectativas_fallidas"] > 0].sort_values(
        "porcentaje_calidad"
    )
    if con_fallas.empty:
        st.success(
            "Todas las particiones evaluadas pasaron el 100% de sus expectativas."
        )
    else:
        st.dataframe(
            con_fallas[
                ["fuente", "fecha", "expectativas_fallidas", "porcentaje_calidad", "filas_evaluadas"]
            ],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# 4) Vista 3: Riesgo de credito (Semana 11, capa Gold)
# ---------------------------------------------------------------------------
def seccion_riesgo_credito() -> None:
    st.header("Vista 3 - Riesgo de credito (Gold)")
    st.caption(
        "De donde sale esto: transformacion_gold_dag entrena un modelo "
        "sobre loan_default_risk y califica a cada cliente de "
        "personal_finance_ml. Ver config/gold_schema.yml para el diseno "
        "completo (por que estas 2 variables, por que se excluyo la 3ra)."
    )

    metadata = _gold_metadata_cacheada()
    df_segmento = _gold_segmento_cacheado()
    df_clientes = _gold_clientes_cacheados()

    if metadata.empty or df_segmento.empty:
        st.info(
            "Todavia no hay tablas Gold. Corre un backfill de "
            "transformacion_gold_dag para ver datos aqui."
        )
        return

    # --- KPIs: que tan bueno es el modelo + cuantos clientes en riesgo ---
    ultima_corrida = metadata.sort_values("fecha").iloc[-1]
    total_clientes = int(df_segmento["num_clientes"].sum())
    clientes_riesgo_alto = (
        int((df_clientes["riesgo_flag"] == "Alto").sum()) if not df_clientes.empty else None
    )
    pct_riesgo_alto = (
        round(clientes_riesgo_alto / total_clientes * 100, 1)
        if clientes_riesgo_alto is not None and total_clientes
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clientes calificados", f"{total_clientes:,}")
    col2.metric("% en riesgo alto", f"{pct_riesgo_alto}%" if pct_riesgo_alto is not None else "N/D")
    col3.metric("Accuracy del modelo", f"{ultima_corrida['accuracy']:.0%}")
    col4.metric("AUC del modelo", f"{ultima_corrida['auc']:.3f}")

    # --- % de riesgo alto por banda de credit score (ordinal) ---
    # Aqui es donde se valida si el modelo tiene sentido de negocio: se
    # espera que el riesgo BAJE conforme el score SUBE. Si esta grafica
    # sale plana o al reves, es señal de que algo esta mal en el modelo,
    # no solo un numero mas.
    riesgo_por_banda = (
        df_segmento.groupby("rango_credit_score")
        .apply(
            lambda g: pd.Series(
                {
                    "clientes": g["num_clientes"].sum(),
                    "riesgo_alto_ponderado": (g["pct_riesgo_alto"] * g["num_clientes"]).sum()
                    / g["num_clientes"].sum(),
                }
            )
        )
        .reset_index()
    )
    riesgo_por_banda["rango_credit_score"] = pd.Categorical(
        riesgo_por_banda["rango_credit_score"], categories=ORDEN_BANDAS, ordered=True
    )
    riesgo_por_banda = riesgo_por_banda.sort_values("rango_credit_score")

    fig_banda = px.bar(
        riesgo_por_banda,
        x="rango_credit_score",
        y="riesgo_alto_ponderado",
        text=riesgo_por_banda["riesgo_alto_ponderado"].round(1).astype(str) + "%",
        labels={"rango_credit_score": "Banda de credit score", "riesgo_alto_ponderado": "% riesgo alto"},
    )
    fig_banda.update_traces(marker_color=COLOR_BANDAS["Excelente"], textposition="outside")
    fig_banda.update_yaxes(range=[0, max(60, riesgo_por_banda["riesgo_alto_ponderado"].max() * 1.2)])
    st.plotly_chart(_tema_oscuro(fig_banda, mostrar_leyenda=False), use_container_width=True)

    # --- Distribucion de clientes por region, segmentada por banda ---
    fig_region = px.bar(
        df_segmento,
        x="region",
        y="num_clientes",
        color="rango_credit_score",
        category_orders={"rango_credit_score": ORDEN_BANDAS},
        color_discrete_map=COLOR_BANDAS,
        barmode="stack",
        labels={"region": "Region", "num_clientes": "Clientes", "rango_credit_score": "Banda"},
    )
    st.plotly_chart(_tema_oscuro(fig_region), use_container_width=True)

    # --- Top 10 clientes de mayor riesgo (para saber a quien revisar primero) ---
    if not df_clientes.empty:
        st.subheader("Top 10 clientes de mayor riesgo")
        columnas = ["user_id", "region", "credit_score", "loan_amount_usd", "riesgo_score", "riesgo_flag"]
        top_riesgo = df_clientes.sort_values("riesgo_score", ascending=False).head(10)
        st.dataframe(top_riesgo[columnas], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 5) Resumen ejecutivo / storytelling (Paso 6)
# ---------------------------------------------------------------------------
def seccion_resumen_ejecutivo() -> None:
    """Interpreta en texto lo que ya calculamos en las dos vistas.

    A proposito NO es texto fijo escrito a mano ("todo va bien!"): cada
    frase se arma con los mismos numeros que ya vimos en las graficas, asi
    que si la proxima semana algo se degrada, el resumen lo dice solo -
    nadie tiene que acordarse de venir a reescribirlo.
    """
    eventos = _eventos_cacheados()
    reportes = _reportes_cacheados()
    gold_metadata = _gold_metadata_cacheada()

    if eventos.empty and reportes.empty and gold_metadata.empty:
        st.info("Corre un backfill para que aparezca el resumen ejecutivo.")
        return

    puntos = []

    if not eventos.empty and "nivel" in eventos.columns:
        total = len(eventos)
        alertas = int((eventos["nivel"] == "ALERTA").sum())
        tasa_exito = round((1 - alertas / total) * 100, 1) if total else 0.0
        if alertas == 0:
            puntos.append(
                f"**Malla estable:** {total} eventos registrados, ninguno en ALERTA "
                f"(100% de tasa de exito)."
            )
        else:
            puntos.append(
                f"**Malla con avisos:** {alertas} de {total} eventos en ALERTA "
                f"({tasa_exito}% de tasa de exito) - revisa la tabla de 'Ultimas alertas'."
            )

    if not reportes.empty:
        calidad_promedio = round(reportes["porcentaje_calidad"].mean(), 1)
        particiones_con_fallas = int((reportes["expectativas_fallidas"] > 0).sum())
        total_particiones = len(reportes)
        fuentes_distintas = reportes["fuente"].nunique()

        if particiones_con_fallas == 0:
            puntos.append(
                f"**Calidad consistente:** {total_particiones} particiones evaluadas "
                f"en {fuentes_distintas} fuentes, todas al {calidad_promedio}% de "
                f"expectativas cumplidas - ninguna por debajo del umbral de "
                f"{UMBRAL_CALIDAD_OK}%."
            )
        else:
            puntos.append(
                f"**Calidad en riesgo:** {particiones_con_fallas} de "
                f"{total_particiones} particiones tienen expectativas fallidas "
                f"(promedio general {calidad_promedio}%) - revisa la tabla de "
                f"'Particiones con expectativas fallidas'."
            )

    if not gold_metadata.empty:
        ultima = gold_metadata.sort_values("fecha").iloc[-1]
        puntos.append(
            f"**Modelo de riesgo:** entrenado sobre {int(ultima['filas_entrenamiento'])} "
            f"casos, accuracy {ultima['accuracy']:.0%} y AUC {ultima['auc']:.3f} en la "
            f"corrida mas reciente ({ultima['fecha']}) - {int(ultima['clientes_calificados']):,} "
            f"clientes calificados."
        )

    st.subheader("Resumen ejecutivo")
    for punto in puntos:
        st.markdown(f"- {punto}")


# ---------------------------------------------------------------------------
# Layout general
# ---------------------------------------------------------------------------
seccion_chequeo_configuracion()
seccion_resumen_ejecutivo()
st.divider()
seccion_salud_malla()
st.divider()
seccion_calidad_datos()
st.divider()
seccion_riesgo_credito()