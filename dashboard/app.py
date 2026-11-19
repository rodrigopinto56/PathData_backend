"""
dashboard/app.py

Semana 10: Dashboard de Calidad - Gemelo Digital Financiero.

Estructura del archivo (de arriba a abajo):
  0) Vista 0: Observabilidad        (Semana 18 - metricas operativas, SLA,
                                     alertas; primera vista del tablero)
  1) Chequeo de configuracion       (Paso 2 - "hola mundo")
  2) Vista 1: Salud de la malla     (Paso 4)
  3) Vista 2: Calidad de datos      (Paso 5)
  4) Vista 3: Riesgo de credito     (Semana 11 - capa Gold)
  5) Vista 4: KPIs financieros      (Semana 12 - riesgo, exposicion,
                                     comportamiento transaccional)
  6) Vista 5: Modelo de riesgo e IA (Semanas 14-15 - dataset de features,
                                     candidatos, calibracion, equidad, PSI)

Cada vista es una funcion aparte para que crezcan sin pisarse entre si.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    import observabilidad as obs
except ModuleNotFoundError:  # ejecucion local fuera del contenedor
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import observabilidad as obs

from data_loader import (
    cargar_eventos_pipeline,
    cargar_reportes_calidad,
    cargar_gold_metadata_modelo,
    cargar_gold_metricas_segmento,
    cargar_gold_clientes_riesgo,
    cargar_gold_kpis,
    cargar_gold_kpis_segmento,
    cargar_historial_modelos,
    cargar_modelo_campeon,
    cargar_tabla_particion,
)

import tema_bbva as tema

# --- Paleta fija del proyecto (misma en toda vista, nunca improvisada) ---
# S19: todos los tokens vienen del tema comun de BBVA (dashboard/tema_bbva.py),
# el mismo de los tableros ejecutivo y del cliente.
SURFACE = tema.SUPERFICIE
TEXTO_PRIMARIO = tema.TEXTO
TEXTO_SECUNDARIO = tema.TEXTO_2
GRIDLINE = tema.LINEA
BASELINE = tema.EJE

# 'nivel' es un estado (INFO = todo bien, ALERTA = algo fallo), por eso usa
# la paleta de "status", no colores categoricos genericos.
COLOR_STATUS = {"INFO": tema.VERDE, "ALERTA": tema.ROJO, "OTRO": tema.GRIS}
COLOR_SERIE_PRINCIPAL = tema.SERIE  # slot 1 categorico (azul medio BBVA)

# 'fuente' si es identidad (dos datasets distintos, sin orden ni jerarquia
# entre ellos), por eso usa colores categoricos en orden fijo (slot 1, slot 3:
# azul y naranja, la pareja con mayor separacion), nunca el verde/rojo de 'nivel'.
COLOR_FUENTE = {"loan_default_risk": tema.MEDIO, "personal_finance_ml": tema.NARANJA}
UMBRAL_CALIDAD_OK = 95.0  # % minimo antes de considerar una particion en riesgo

# 'rango_credit_score' SI tiene un orden (Bajo -> Excelente), por eso usa
# una rampa secuencial de un solo tono (mas oscuro = mejor score) en vez de
# colores categoricos sin relacion entre si -- ver gold_schema.yml.
ORDEN_BANDAS = ["Bajo", "Regular", "Bueno", "Muy bueno", "Excelente"]
COLOR_BANDAS = dict(zip(ORDEN_BANDAS, tema.SECUENCIAL))
COLOR_RIESGO = {"Alto": tema.ROJO, "Bajo": tema.VERDE}  # mismo par que 'nivel': es un status

# Vista 4: 'estado' de un KPI es un status (mismo verde/rojo de siempre);
# INFO = KPI sin umbral, solo informativo.
COLOR_ESTADO_KPI = {"OK": tema.VERDE, "ALERTA": tema.ROJO, "INFO": tema.GRIS}
# Semaforo del indice de riesgo consolidado: es un status ordenado
# (verde -> ambar -> rojo), no categorias sin relacion.
ORDEN_SEMAFORO = ["Bajo", "Medio", "Alto"]
COLOR_SEMAFORO = {"Bajo": tema.VERDE, "Medio": tema.AMBAR, "Alto": tema.ROJO}
# Exposicion vs perdida esperada: dos medidas distintas en la misma
# unidad (USD) -> categoricos slot 1 y slot 3.
COLOR_MEDIDA_USD = {"Exposicion": tema.MEDIO, "Perdida esperada": tema.NARANJA}
ORDEN_PERFIL_AHORRO = ["Deficitario", "Equilibrado", "Ahorrador", "Sin ingreso"]
COLOR_PERFIL_AHORRO = {
    "Deficitario": tema.ROJO,
    "Equilibrado": "#B8C2CE",
    "Ahorrador": tema.VERDE,
    "Sin ingreso": "#5B6878",
}
# Nombre legible de cada KPI (las llaves son los ids de kpis_financieros.py)
NOMBRE_KPI = {
    "capacidad_ahorro_mediana_pct": "Capacidad de ahorro (mediana)",
    "capacidad_ahorro_ic95_inf_pct": "Capacidad de ahorro · IC 95% inferior",
    "capacidad_ahorro_ic95_sup_pct": "Capacidad de ahorro · IC 95% superior",
    "endeudamiento_mensual_mediana_pct": "Endeudamiento mensual (mediana)",
    "pct_clientes_endeudamiento_alto": "Clientes con endeudamiento alto",
    "varianza_ingresos_gastos": "Varianza ingresos vs. gastos",
    "cv_ingreso_6m_pct": "Variación del ingreso (6 meses)",
    "cv_gasto_6m_pct": "Variación del gasto (6 meses)",
    "indice_riesgo_consolidado_promedio": "Índice de riesgo consolidado",
    "pct_clientes_semaforo_bajo": "Clientes en verde",
    "pct_clientes_semaforo_medio": "Clientes en amarillo",
    "pct_clientes_semaforo_alto": "Clientes en rojo",
    "tasa_incumplimiento_historica_pct": "Incumplimiento histórico",
    "probabilidad_incumplimiento_promedio_pct": "Probabilidad de impago promedio",
    "pct_clientes_riesgo_alto": "Clientes de riesgo alto",
    "credit_score_promedio": "Credit score promedio",
    "pct_clientes_subprime": "Clientes subprime",
    "dti_mediana": "Deuda / ingreso (mediana)",
    "auc_modelo_riesgo": "AUC del modelo de riesgo",
    "num_prestamos_activos": "Préstamos activos",
    "pct_clientes_con_prestamo": "Clientes con préstamo",
    "exposicion_total_usd": "Exposición total",
    "exposicion_promedio_usd": "Exposición promedio",
    "exposicion_riesgo_alto_usd": "Exposición en riesgo alto",
    "pct_exposicion_riesgo_alto": "% de exposición en riesgo alto",
    "perdida_esperada_usd": "Pérdida esperada",
    "perdida_esperada_pct_exposicion": "Pérdida esperada / exposición",
    "indice_concentracion_region_hhi": "Concentración por región (HHI)",
    "pct_exposicion_region_principal": "Exposición en la región principal",
    "ingreso_mensual_promedio_usd": "Ingreso mensual promedio",
    "gasto_mensual_promedio_usd": "Gasto mensual promedio",
    "ratio_gasto_ingreso_mediana": "Gasto / ingreso (mediana)",
    "tasa_ahorro_mediana_pct": "Tasa de ahorro (mediana)",
    "flujo_libre_mensual_mediana_usd": "Flujo libre mensual (mediana)",
    "pct_clientes_deficitarios": "Clientes deficitarios",
    "meses_cobertura_ahorro_mediana": "Meses cubiertos con ahorro",
    "pct_clientes_sin_colchon": "Clientes sin fondo de emergencia",
}


def _nombre_kpi(kpi: str) -> str:
    return NOMBRE_KPI.get(kpi, kpi.replace("_", " ").capitalize())


TITULOS_CATEGORIA_KPI = {
    "kpis_negocio": "KPIs de negocio",
    "riesgo_financiero": "Riesgo financiero",
    "exposicion": "Exposición",
    "comportamiento_transaccional": "Comportamiento transaccional",
}

tema.configurar_pagina("Dashboard técnico", ancho_max=1320)
tema.encabezado(
    "Dashboard técnico", eyebrow="Gemelo Digital Financiero · Plataforma de datos",
    subtitulo="Observabilidad, calidad de datos, riesgo de crédito, KPIs y modelo de IA en un solo lugar.",
    meta_html="Arquitectura Medallion<br><b>Bronze → Silver → Gold</b>",
)


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


@st.cache_data(ttl=30)
def _gold_kpis_cacheados() -> pd.DataFrame:
    return cargar_gold_kpis()


@st.cache_data(ttl=30)
def _gold_kpis_segmento_cacheados() -> pd.DataFrame:
    return cargar_gold_kpis_segmento()


def _estilo(fig, mostrar_leyenda: bool = True):
    """Mismo fondo, tipografia, rejilla y colores que el resto de los tableros
    (tema BBVA). Sin esto, Plotly usa su estilo por default."""
    return tema.estilo_plotly(fig, mostrar_leyenda=mostrar_leyenda)


# ---------------------------------------------------------------------------
# 0) Vista 0: Observabilidad de la plataforma (Semana 18)
# ---------------------------------------------------------------------------
# Estado de un SLA / de la plataforma: paleta de status (icono + texto,
# nunca solo color). Estado de una corrida de Airflow: tambien es status.
ICONO_SLA = {obs.OK: "✓", obs.ALERTA: "!", obs.SIN_DATOS: "–"}
COLOR_SLA = {obs.OK: tema.VERDE, obs.ALERTA: tema.ROJO, obs.SIN_DATOS: tema.GRIS}
TEXTO_SLA = {obs.OK: "Cumple", obs.ALERTA: "No cumple", obs.SIN_DATOS: "Sin datos"}
COLOR_GENERAL = {"Saludable": tema.VERDE, "Requiere atención": tema.AMBAR, "Crítico": tema.ROJO,
                 "Sin datos": tema.GRIS}
ICONO_GENERAL = {"Saludable": "✓", "Requiere atención": "!", "Crítico": "✕", "Sin datos": "–"}
COLOR_CORRIDA = {"success": tema.VERDE, "failed": tema.ROJO, "running": COLOR_SERIE_PRINCIPAL,
                 "queued": tema.GRIS, "upstream_failed": tema.AMBAR}
SIMBOLO_OPERADOR = {">=": "≥", "<=": "≤"}
NOMBRE_ESTADO_CORRIDA = {"success": "Exitosa", "failed": "Fallida", "running": "En curso", "queued": "En cola"}


@st.cache_data(ttl=60)
def _historial_modelos_obs() -> pd.DataFrame:
    return cargar_historial_modelos()


def _cargar_observabilidad() -> tuple[dict, dict, pd.Timestamp]:
    """Junta las fuentes de la Vista 0 (sin cache: el estado debe ser el de ahora)."""
    config = obs.cargar_config()
    zona = config["zona_horaria"]
    ahora = pd.Timestamp.now(tz=zona)
    entradas = obs.cargar_entradas_locales()
    entradas["corridas"] = obs.preparar_corridas(entradas["corridas"], zona)
    entradas["calidad"] = _reportes_cacheados()
    historial = _historial_modelos_obs()
    entradas["modelo"] = {
        "fecha_entrenamiento": historial["entrenado_en"].dropna().max() if not historial.empty else None,
    }
    return entradas, config, ahora


def _tarjeta_sla(s: pd.Series) -> str:
    color = COLOR_SLA[s["estado"]]
    valor = obs._fmt(s["valor"], s["unidad"])
    meta = f"Meta {SIMBOLO_OPERADOR.get(s['operador'], s['operador'])} {obs._fmt(s['meta'], s['unidad'])}"
    nota = (f"<div style='color:{tema.TEXTO_3};font-size:0.75rem;margin-top:6px'>{s['detalle']}</div>"
            if s["detalle"] else "")
    return (
        f"<div style='background:{tema.SUPERFICIE};border:1px solid {tema.LINEA};border-top:4px solid {color};"
        f"border-radius:12px;padding:14px 16px;margin-bottom:12px;min-height:150px;box-shadow:0 1px 2px rgba(7,33,70,0.04)'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:8px'>"
        f"<div style='font-size:0.74rem;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;color:{tema.TEXTO_2}'>"
        f"{s['nombre']}</div>{tema.chip(TEXTO_SLA[s['estado']], color, ICONO_SLA[s['estado']])}</div>"
        f"<div style='font-size:1.7rem;font-weight:600;color:{tema.TEXTO};margin-top:8px;"
        f"font-variant-numeric:tabular-nums'>{valor}</div>"
        f"<div style='font-size:0.8rem;color:{tema.TEXTO_2}'>{meta}</div>"
        f"{nota}</div>"
    )


def seccion_observabilidad() -> None:
    tema.seccion("Observabilidad de la plataforma", eyebrow="Vista 0 · Operación", primera=True)
    try:
        entradas, config, ahora = _cargar_observabilidad()
    except FileNotFoundError as error:
        st.error(f"No se encontró la configuración de SLA: {error}")
        return

    exportacion = entradas.get("exportacion") or {}
    if exportacion:
        generado = pd.Timestamp(exportacion["generado_en"]).tz_convert(ahora.tz)
        minutos = int((ahora - generado).total_seconds() // 60)
        st.caption(
            f"Métricas operativas, SLA ({config['ventana_dias']} días) y alertas. Corridas de Airflow exportadas por "
            f"monitoreo_dag el {generado:%d/%m %H:%M} (hace {minutos} min); el resto se lee en vivo."
        )
        if minutos > 120:
            st.warning("La exportación de Airflow tiene más de 2 horas: revisa que monitoreo_dag esté activo.")
    else:
        st.info(
            "Todavía no hay métricas de Airflow. Activa **monitoreo_dag** en Airflow (corre cada hora) o "
            "disparalo a mano; mientras tanto se muestran los SLA que no dependen de Airflow."
        )

    corridas = entradas["corridas"]
    slas = obs.evaluar_slas(entradas, config, ahora)
    alertas = obs.alertas_activas(slas, corridas, entradas["eventos"], ahora, tareas=entradas["tareas"])
    general = obs.estado_general(slas, alertas)

    # --- Resumen en una linea ---
    monitoreadas = corridas[corridas["dag_id"].isin(config["dags_monitoreados"])] if not corridas.empty else corridas
    recientes = obs.en_ventana(monitoreadas, ahora, config["ventana_dias"]) if not monitoreadas.empty else monitoreadas
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div style='font-size:0.8rem;font-weight:600;color:{tema.TEXTO_2}'>Estado de la plataforma</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{COLOR_GENERAL[general]}'>"
        f"{ICONO_GENERAL[general]} {general}</div>", unsafe_allow_html=True)
    con_datos = slas[slas["estado"] != obs.SIN_DATOS]
    c2.metric("SLA cumplidos", f"{int((slas['estado'] == obs.OK).sum())} de {len(con_datos)}",
              help="Solo cuentan los SLA con datos suficientes.")
    c3.metric("Alertas abiertas", len(alertas))
    c4.metric(f"Corridas ({config['ventana_dias']} días)", len(recientes))

    # --- SLA ---
    st.subheader("Acuerdos de nivel de servicio (SLA)")
    for inicio in range(0, len(slas), 4):
        columnas = st.columns(4)
        for columna, (_, s) in zip(columnas, slas.iloc[inicio:inicio + 4].iterrows()):
            columna.markdown(_tarjeta_sla(s), unsafe_allow_html=True)
    with st.expander("¿Cómo se mide cada SLA?"):
        st.dataframe(slas[["nombre", "descripcion", "operador", "meta", "unidad", "severidad"]]
                     .rename(columns={"nombre": "SLA", "descripcion": "Cómo se mide", "operador": "Regla",
                                      "meta": "Meta", "unidad": "Unidad", "severidad": "Severidad"}),
                     hide_index=True, use_container_width=True)
        st.caption("Configuración: config/slas.yml. Los mismos cálculos los hace monitoreo_dag, que registra "
                   "un evento en el log cuando un SLA cambia de estado.")

    # --- Alertas ---
    st.subheader("Alertas abiertas")
    if alertas.empty:
        st.success("Sin alertas: todos los SLA con datos se cumplen y no hay corridas ni tareas fallidas sin resolver.")
    else:
        vista = alertas.copy()
        vista["severidad"] = vista["severidad"].map({"alta": "! Alta", "media": "· Media"}).fillna(vista["severidad"])
        vista["desde"] = vista["desde"].apply(lambda t: "" if pd.isna(t) else f"{t:%d/%m %H:%M}")
        st.dataframe(vista.rename(columns={"severidad": "Severidad", "origen": "Origen", "elemento": "Elemento",
                                           "mensaje": "Detalle", "desde": "Desde"}),
                     hide_index=True, use_container_width=True)

    tab_corridas, tab_tendencias, tab_datos, tab_asistente = st.tabs(
        ["Corridas", "Tendencias", "Calidad y modelo", "Asistente IA"])

    with tab_corridas:
        if corridas.empty:
            st.info("Sin corridas exportadas todavía.")
        else:
            ultimas = corridas[corridas["start_date"] >= ahora - pd.Timedelta(days=3)].copy()
            # En curso -> hasta ahora; fin anterior al inicio (corrida re-ejecutada) -> barra minima
            ultimas["fin"] = ultimas["end_date"].fillna(ahora)
            ultimas["fin"] = ultimas["fin"].where(ultimas["fin"] >= ultimas["start_date"],
                                                  ultimas["start_date"] + pd.Timedelta(minutes=1))
            ultimas["Estado"] = ultimas["state"].map(NOMBRE_ESTADO_CORRIDA).fillna(ultimas["state"])
            if not ultimas.empty:
                fig = px.timeline(
                    ultimas, x_start="start_date", x_end="fin", y="dag_id", color="state",
                    color_discrete_map=COLOR_CORRIDA,
                    hover_data={"run_id": True, "Estado": True, "duracion_min": ":.1f", "state": False,
                                "start_date": "|%d/%m %H:%M", "fin": "|%d/%m %H:%M", "dag_id": False},
                    labels={"dag_id": "", "duracion_min": "Duración (min)", "start_date": "Inicio", "fin": "Fin"},
                )
                fig.for_each_trace(lambda t: t.update(name=NOMBRE_ESTADO_CORRIDA.get(t.name, t.name)))
                fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(config["dags_monitoreados"])))
                fig.update_layout(height=280, legend=dict(orientation="h", y=1.12, x=0))
                st.plotly_chart(_estilo(fig), use_container_width=True)
                st.caption("Últimos 3 días. Cada barra es una corrida: de cuando empezó a cuando terminó.")
            resumen = obs.resumen_por_dag(obs.en_ventana(corridas, ahora, config["ventana_dias"]))
            if not resumen.empty:
                resumen["ultima_corrida"] = resumen["ultima_corrida"].dt.strftime("%d/%m %H:%M")
                st.dataframe(resumen.rename(columns={
                    "dag_id": "DAG", "corridas": "Corridas", "exitosas": "Exitosas", "fallidas": "Fallidas",
                    "en_curso": "En curso / en fila",
                    "tasa_exito_pct": "% exito", "duracion_mediana_min": "Duracion tipica (min)",
                    "duracion_p95_min": "Duracion p95 (min)", "ultima_corrida": "Ultima", "ultimo_estado": "Estado"}),
                    hide_index=True, use_container_width=True)

    with tab_tendencias:
        exitosas = corridas[(corridas["state"] == "success") & corridas["dag_id"].isin(config["dags_monitoreados"])] \
            if not corridas.empty else corridas
        if exitosas.empty:
            st.info("Sin corridas exitosas para graficar.")
        else:
            st.markdown("**Duración de las corridas exitosas (minutos)**")
            fig = px.line(exitosas.sort_values("start_date"), x="start_date", y="duracion_min",
                          facet_col="dag_id", facet_col_wrap=1, facet_row_spacing=0.12, markers=True,
                          labels={"start_date": "", "duracion_min": "min"},
                          category_orders={"dag_id": config["dags_monitoreados"]})
            fig.update_traces(line=dict(color=COLOR_SERIE_PRINCIPAL, width=2), marker=dict(size=7))
            # Nombre de cada DAG como titulo horizontal arriba a la izquierda de su panel
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_color=TEXTO_SECUNDARIO,
                                                       x=0, xanchor="left"))
            # Escala logaritmica: una corrida que quedo a medias (laptop suspendida, ~800 min) no
            # aplasta a las normales (~1-10 min); ambas se ven.
            fig.update_yaxes(matches=None, title="", showticklabels=True, type="log")
            fig.update_xaxes(showticklabels=True)
            fig.update_layout(height=150 * exitosas["dag_id"].nunique() + 40, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(_estilo(fig, mostrar_leyenda=False), use_container_width=True)
            st.caption("Un panel por DAG, en escala logarítmica (cada línea de la cuadrícula es x10): lo que importa "
                       "es si la duración crece. Un pico aislado suele ser una corrida interrumpida (laptop suspendida).")

            sla_frescura = next(s for s in config["slas"] if s["metrica"] == "frescura_gold_pct")
            frescura = obs.frescura_por_dia(corridas, "transformacion_gold_dag", sla_frescura["hora_limite"],
                                            ahora, config["historial_dias"])
            if not frescura.empty:
                st.markdown(f"**Frescura: ¿Gold listo antes de las {sla_frescura['hora_limite']} del día siguiente?**")
                tabla = pd.DataFrame({
                    "Particion": frescura["particion"].dt.strftime("%Y-%m-%d"),
                    "Listo": frescura["listo_en"].apply(lambda t: "No corrio" if pd.isna(t) else f"{t:%d/%m %H:%M}"),
                    "Limite": frescura["limite"].dt.strftime("%d/%m %H:%M"),
                    "A tiempo": frescura["a_tiempo"].map({True: "✓ Si", False: "✕ No"}),
                    "Retraso (min)": [("" if a else f"{r:,.0f}") for a, r in zip(frescura["a_tiempo"], frescura["retraso_min"].fillna(0))],
                }).iloc[::-1]
                st.dataframe(tabla, hide_index=True, use_container_width=True, height=260)

    with tab_datos:
        calidad = entradas["calidad"]
        if calidad is None or calidad.empty:
            st.info("Sin reportes de calidad de Silver.")
        else:
            ultimos = calidad.sort_values("fecha").groupby("fuente").tail(1)
            fig = px.bar(ultimos.sort_values("porcentaje_calidad"), x="porcentaje_calidad", y="fuente",
                         orientation="h", text="porcentaje_calidad",
                         labels={"porcentaje_calidad": "% de reglas cumplidas", "fuente": ""})
            fig.update_traces(marker_color=COLOR_SERIE_PRINCIPAL, texttemplate="%{text:.1f}%", textposition="outside")
            sla_calidad = next(s for s in config["slas"] if s["metrica"] == "calidad_silver_min_pct")
            fig.add_vline(x=sla_calidad["meta"], line_dash="dot", line_color=TEXTO_SECUNDARIO,
                          annotation_text=f"Meta {sla_calidad['meta']}%", annotation_font_color=TEXTO_SECUNDARIO)
            fig.update_layout(height=60 * len(ultimos) + 80, xaxis_range=[0, 105])
            st.markdown("**Calidad de Silver: último reporte de cada fuente**")
            st.plotly_chart(_estilo(fig, mostrar_leyenda=False), use_container_width=True)
        historial = _historial_modelos_obs()
        if not historial.empty:
            ultimo = historial.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("Último entrenamiento", str(ultimo["entrenado_en"])[:16].replace("T", " "))
            m2.metric("Versiones entrenadas", len(historial))
            m3.metric("AUC de la última versión", f"{ultimo['auc']:.3f}" if pd.notna(ultimo["auc"]) else "N/D")

    with tab_asistente:
        metricas = obs.metricas_asistente(entradas["asistente_log"], ahora, config["ventana_dias"])
        evaluacion = entradas["evaluacion"]
        evaluacion_asesor = entradas.get("evaluacion_asesor") or {}
        if metricas["preguntas"] == 0 and not evaluacion and not evaluacion_asesor:
            st.info("Todavía no hay preguntas registradas ni evaluaciones del asistente.")
        if metricas["preguntas"]:
            canales = metricas.get("por_canal", {})
            a1, a2, a3, a4 = st.columns(4)
            a1.metric(f"Preguntas ({config['ventana_dias']} días)", metricas["preguntas"],
                      help=f"Ejecutivos: {canales.get('ejecutivo', 0)} · Clientes (asesor): {canales.get('cliente', 0)} · "
                           f"Derivadas a una persona (fraude, privacidad): {metricas.get('derivadas', 0)}")
            a2.metric("Redactadas por la IA", f"{metricas['pct_ia']:.0f}%",
                      help=f"Respaldo (datos exactos): {metricas['pct_respaldo']:.0f}% · Modo básico: {metricas['pct_basico']:.0f}%")
            a3.metric("Tiempo típico", f"{metricas['segundos_mediana']:.0f} s")
            a4.metric("Tiempo p95", f"{metricas['segundos_p95']:.0f} s")
        if evaluacion:
            st.markdown(
                f"**Última evaluación automática** ({evaluacion.get('modelo', '')}, "
                f"{str(evaluacion.get('evaluado_en', ''))[:16].replace('T', ' ')}): "
                f"éxito final **{evaluacion.get('exito_final_pct')}%**, veracidad final "
                f"**{evaluacion.get('veracidad_final_pct')}%**, {evaluacion.get('preguntas')} preguntas "
                f"(meta {evaluacion.get('meta_pct', 84)}%)."
            )
        if evaluacion_asesor:
            st.markdown(
                f"**Última evaluación del asesor del cliente** ({evaluacion_asesor.get('modelo', '')}, "
                f"{str(evaluacion_asesor.get('evaluado_en', ''))[:16].replace('T', ' ')}): "
                f"éxito final **{evaluacion_asesor.get('exito_final_pct')}%**, veracidad final "
                f"**{evaluacion_asesor.get('veracidad_final_pct')}%**, {evaluacion_asesor.get('preguntas')} preguntas "
                f"(meta {evaluacion_asesor.get('meta_pct', 84)}%)."
            )
        log = entradas["asistente_log"]
        if not log.empty:
            log = log.assign(_t=pd.to_datetime(log["timestamp"], utc=True, format="ISO8601"))
            recientes_log = log.sort_values("_t", ascending=False).head(10)
            columnas = [c for c in ["timestamp", "canal", "pregunta", "modo", "segundos", "reintentos"] if c in recientes_log.columns]
            st.markdown("**Últimas preguntas**")
            st.dataframe(recientes_log[columnas].assign(
                timestamp=pd.to_datetime(recientes_log["timestamp"], utc=True, format="ISO8601")
                .dt.tz_convert(ahora.tz).dt.strftime("%d/%m %H:%M")),
                hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# 1) Chequeo de configuracion (Paso 2)
# ---------------------------------------------------------------------------
def seccion_chequeo_configuracion() -> None:
    with st.expander("Chequeo de configuración", expanded=False):
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
    tema.seccion("Salud de la malla", eyebrow="Vista 1 · Pipelines", primera=True)
    st.caption(
        "¿De dónde sale esto? cada tarea de tus DAGs (ingesta_bronze_dag y "
        "transformacion_silver_dag) escribe un evento en pipeline_log.jsonl "
        "cuando corre. Aquí se cuentan esos eventos."
    )

    eventos = _eventos_cacheados().copy()

    if eventos.empty:
        st.info(
            "Todavía no hay eventos registrados. Corre un backfill de "
            "transformacion_silver_dag para ver datos aquí."
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
                f"({', '.join(sorted(valores_inesperados))}), agrupados aquí como 'OTRO'."
            )

    # --- KPIs (numeros grandes, primero, para responder rapido "vamos bien?") ---
    total_eventos = len(eventos)
    eventos_alerta = int((eventos["nivel"] == "ALERTA").sum()) if "nivel" in eventos else 0
    tasa_exito = round((1 - eventos_alerta / total_eventos) * 100, 1) if total_eventos else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Eventos totales", f"{total_eventos:,}")
    col2.metric("Tasa de éxito", f"{tasa_exito}%")
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
        st.plotly_chart(_estilo(fig), use_container_width=True)

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
        st.plotly_chart(_estilo(fig_tareas, mostrar_leyenda=False), use_container_width=True)

    # --- Tabla de las alertas mas recientes (para saber que revisar primero) ---
    if eventos_alerta:
        st.subheader("Últimas alertas")
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
    tema.seccion("Calidad de datos", eyebrow="Vista 2 · Great Expectations", primera=True)
    st.caption(
        "¿De dónde sale esto? cada partición (fuente + fecha) que pasa por "
        "transformacion_silver_dag genera un _calidad_reporte.json en MinIO, "
        "con el resultado de las expectativas de Great Expectations definidas "
        "en config/silver_schema.yml."
    )

    reportes = _reportes_cacheados()

    if reportes.empty:
        st.info(
            "Todavía no hay reportes de calidad en MinIO. Corre un backfill "
            "de transformacion_silver_dag para ver datos aquí."
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
            "fecha": "Partición (fecha)",
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
    st.plotly_chart(_estilo(fig), use_container_width=True)

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
    tema.seccion("Riesgo de crédito", eyebrow="Vista 3 · Capa Gold", primera=True)
    st.caption(
        "¿De dónde sale esto? transformacion_gold_dag entrena un modelo "
        "sobre loan_default_risk y califica a cada cliente de "
        "personal_finance_ml. Ver config/gold_schema.yml para el diseno "
        "completo (por que estas 2 variables, por que se excluyo la 3ra)."
    )

    metadata = _gold_metadata_cacheada()
    df_segmento = _gold_segmento_cacheado()
    df_clientes = _gold_clientes_cacheados()

    if metadata.empty or df_segmento.empty:
        st.info(
            "Todavía no hay tablas Gold. Corre un backfill de "
            "transformacion_gold_dag para ver datos aquí."
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
    # El modelo S15 reporta Gini (estandar de riesgos); el de S11, accuracy
    if pd.notna(ultima_corrida.get("gini")):
        col3.metric("Gini del modelo", f"{ultima_corrida['gini']:.2f}", help=f"Modelo {ultima_corrida.get('version')}")
    else:
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
    st.plotly_chart(_estilo(fig_banda, mostrar_leyenda=False), use_container_width=True)

    # --- Distribucion de clientes por region, segmentada por banda ---
    fig_region = px.bar(
        df_segmento,
        x="region",
        y="num_clientes",
        color="rango_credit_score",
        category_orders={"rango_credit_score": ORDEN_BANDAS},
        color_discrete_map=COLOR_BANDAS,
        barmode="stack",
        labels={"region": "Región", "num_clientes": "Clientes", "rango_credit_score": "Banda"},
    )
    st.plotly_chart(_estilo(fig_region), use_container_width=True)

    # --- Top 10 clientes de mayor riesgo (para saber a quien revisar primero) ---
    if not df_clientes.empty:
        st.subheader("Top 10 clientes de mayor riesgo")
        columnas = ["user_id", "region", "credit_score", "loan_amount_usd", "riesgo_score", "riesgo_flag"]
        top_riesgo = df_clientes.sort_values("riesgo_score", ascending=False).head(10)
        st.dataframe(
            top_riesgo[columnas], use_container_width=True, hide_index=True,
            column_config={
                "user_id": "Cliente", "region": "Región",
                "credit_score": st.column_config.NumberColumn("Credit score", format="%d"),
                "loan_amount_usd": st.column_config.NumberColumn("Préstamo (USD)", format="$%.0f"),
                "riesgo_score": st.column_config.ProgressColumn("Probabilidad de impago", min_value=0, max_value=1,
                                                                format="%.2f"),
                "riesgo_flag": "Nivel de riesgo",
            },
        )


# ---------------------------------------------------------------------------
# 5) Vista 4: KPIs financieros (Semana 12)
# ---------------------------------------------------------------------------
def _formatear_kpi(valor, unidad: str) -> str:
    if valor is None or pd.isna(valor):
        return "N/D"
    if unidad == "USD":
        if abs(valor) >= 1_000_000:
            return f"${valor / 1_000_000:,.1f} M"
        return f"${valor:,.0f}"
    if unidad == "%":
        return f"{valor:.1f}%"
    if unidad == "indice 0-1":
        return f"{valor:.3f}"
    if unidad == "ratio":
        return f"{valor:.2f}"
    if unidad == "puntos 0-100":
        return f"{valor:.1f} / 100"
    if unidad == "meses":
        return f"{valor:.1f} meses"
    return f"{valor:,.0f}"


def seccion_kpis_financieros() -> None:
    tema.seccion("KPIs financieros", eyebrow="Vista 4 · Capa Gold", primera=True)
    st.caption(
        "¿De dónde sale esto? la tarea calcular_kpis de transformacion_gold_dag "
        "agrega fact_posicion_financiera (modelo estrella de Gold) en tres "
        "familias de KPIs. Formulas y umbrales en config/gold_schema.yml, "
        "seccion metricas_financieras."
    )

    kpis = _gold_kpis_cacheados()
    segmentos = _gold_kpis_segmento_cacheados()

    if kpis.empty:
        st.info(
            "Todavía no hay KPIs en Gold. Corre transformacion_gold_dag "
            "(tarea calcular_kpis) para ver datos aquí."
        )
        return

    fechas = sorted(kpis["fecha"].unique(), reverse=True)
    fecha = st.selectbox("Partición", fechas, index=0, key="fecha_kpis")
    kpis_fecha = kpis[kpis["fecha"] == fecha]

    # --- Semaforo: primero lo que requiere accion ---
    en_alerta = kpis_fecha[kpis_fecha["estado"] == "ALERTA"]
    con_umbral = kpis_fecha[kpis_fecha["estado"].isin(["OK", "ALERTA"])]
    if en_alerta.empty:
        st.success(f"Los {len(con_umbral)} KPIs con umbral están dentro de rango.")
    else:
        st.warning(
            f"{len(en_alerta)} de {len(con_umbral)} KPIs con umbral están en ALERTA: "
            + ", ".join(_nombre_kpi(k) for k in en_alerta["kpi"])
        )

    # --- Numeros grandes por familia ---
    # El delta muestra la distancia al umbral (no un cambio vs. ayer):
    # rojo = del lado malo del umbral, verde = del lado bueno.
    for categoria, titulo in TITULOS_CATEGORIA_KPI.items():
        st.markdown(f"### {titulo}")
        del_grupo = kpis_fecha[kpis_fecha["categoria"] == categoria]
        columnas = st.columns(4)
        for i, fila in enumerate(del_grupo.itertuples()):
            delta = None
            if fila.estado in ("OK", "ALERTA") and pd.notna(fila.umbral):
                delta = f"{'Fuera de' if fila.estado == 'ALERTA' else 'Dentro de'} meta ({_formatear_kpi(fila.umbral, fila.unidad)})"
            elif fila.estado == "SIN_DATOS":
                delta = "Sin datos"
            avance = getattr(fila, "avance_meta_pct", None)
            ayuda = fila.descripcion + (f" · Avance hacia la meta: {avance:.0f}%" if avance is not None and pd.notna(avance) else "")
            columnas[i % 4].metric(
                _nombre_kpi(fila.kpi),
                _formatear_kpi(fila.valor, fila.unidad),
                delta=delta,
                delta_color={"ALERTA": "inverse", "SIN_DATOS": "off"}.get(fila.estado, "normal"),
                help=ayuda,
            )

    if segmentos.empty:
        return
    seg_fecha = segmentos[segmentos["fecha"] == fecha]

    por_semaforo = seg_fecha[
        (seg_fecha["dimension"] == "nivel_riesgo_consolidado") & seg_fecha["segmento"].isin(ORDEN_SEMAFORO)
    ].copy()
    if not por_semaforo.empty:
        st.subheader("Semáforo del índice de riesgo consolidado")
        por_semaforo["pct"] = por_semaforo["num_clientes"] / por_semaforo["num_clientes"].sum() * 100
        por_semaforo["cartera"] = "Clientes"
        fig_semaforo = px.bar(
            por_semaforo,
            x="pct",
            y="cartera",
            color="segmento",
            orientation="h",
            category_orders={"segmento": ORDEN_SEMAFORO},
            color_discrete_map=COLOR_SEMAFORO,
            text=por_semaforo["pct"].round(1).astype(str) + "%",
            labels={"pct": "% de clientes", "cartera": "", "segmento": "Riesgo"},
        )
        fig_semaforo.update_layout(height=180)
        fig_semaforo.update_xaxes(range=[0, 100])
        st.plotly_chart(_estilo(fig_semaforo), use_container_width=True)

    # --- Exposicion vs perdida esperada por region ---
    # Barras agrupadas: misma unidad (USD), se compara cuanto de lo
    # prestado en cada region se espera perder.
    por_region = seg_fecha[seg_fecha["dimension"] == "region"].melt(
        id_vars="segmento",
        value_vars=["exposicion_usd", "perdida_esperada_usd"],
        var_name="medida",
        value_name="usd",
    )
    por_region["medida"] = por_region["medida"].map(
        {"exposicion_usd": "Exposicion", "perdida_esperada_usd": "Perdida esperada"}
    )
    st.subheader("Exposición y pérdida esperada por región")
    fig_region = px.bar(
        por_region,
        x="segmento",
        y="usd",
        color="medida",
        barmode="group",
        color_discrete_map=COLOR_MEDIDA_USD,
        labels={"segmento": "Región", "usd": "USD", "medida": ""},
    )
    st.plotly_chart(_estilo(fig_region), use_container_width=True)

    # --- Riesgo por banda de credit score (validacion de negocio) ---
    por_banda = seg_fecha[seg_fecha["dimension"] == "rango_credit_score"].copy()
    por_banda = por_banda[por_banda["segmento"].isin(ORDEN_BANDAS)]
    por_banda["segmento"] = pd.Categorical(por_banda["segmento"], categories=ORDEN_BANDAS, ordered=True)
    por_banda = por_banda.sort_values("segmento")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pérdida esperada / exposición por banda")
        fig_banda = px.bar(
            por_banda,
            x="segmento",
            y="perdida_esperada_pct_exposicion",
            text=por_banda["perdida_esperada_pct_exposicion"].round(1).astype(str) + "%",
            labels={"segmento": "Banda de credit score", "perdida_esperada_pct_exposicion": "% de la exposición"},
        )
        fig_banda.update_traces(marker_color=COLOR_BANDAS["Excelente"], textposition="outside")
        st.plotly_chart(_estilo(fig_banda, mostrar_leyenda=False), use_container_width=True)

    # --- Comportamiento transaccional: perfil de ahorro ---
    with col2:
        st.subheader("Clientes por perfil de ahorro")
        por_perfil = seg_fecha[seg_fecha["dimension"] == "perfil_ahorro"]
        fig_perfil = px.bar(
            por_perfil,
            x="segmento",
            y="num_clientes",
            color="segmento",
            category_orders={"segmento": ORDEN_PERFIL_AHORRO},
            color_discrete_map=COLOR_PERFIL_AHORRO,
            text="num_clientes",
            labels={"segmento": "Perfil", "num_clientes": "Clientes"},
        )
        fig_perfil.update_traces(textposition="outside")
        st.plotly_chart(_estilo(fig_perfil, mostrar_leyenda=False), use_container_width=True)

    # --- Detalle por segmento (para bajar al "donde") ---
    st.subheader("KPIs por segmento")
    dimension = st.selectbox(
        "Dimensión",
        sorted(seg_fecha["dimension"].unique()),
        key="dimension_kpis",
    )
    columnas_tabla = [
        "segmento", "num_clientes", "indice_riesgo_promedio", "pct_semaforo_alto",
        "endeudamiento_mensual_mediana_pct", "tasa_ahorro_ic95_inf_pct", "tasa_ahorro_ic95_sup_pct",
        "pct_riesgo_alto", "pd_promedio_pct",
        "exposicion_usd", "pct_exposicion_del_total", "perdida_esperada_usd",
        "tasa_ahorro_mediana_pct", "pct_deficitarios",
    ]
    # Particiones calculadas con una version anterior no traen todas las
    # columnas: se muestran solo las que existen en vez de tronar.
    columnas_tabla = [c for c in columnas_tabla if c in seg_fecha.columns]
    st.dataframe(
        seg_fecha[seg_fecha["dimension"] == dimension][columnas_tabla].sort_values(
            "exposicion_usd", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# 6) Resumen ejecutivo / storytelling (Paso 6)
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
    kpis = _gold_kpis_cacheados()

    if eventos.empty and reportes.empty and gold_metadata.empty and kpis.empty:
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
        calidad = (
            f"Gini {ultima['gini']:.2f}" if pd.notna(ultima.get("gini")) else f"accuracy {ultima['accuracy']:.0%}"
        )
        puntos.append(
            f"**Modelo de riesgo{' ' + str(ultima['version']) if pd.notna(ultima.get('version')) else ''}:** "
            f"entrenado sobre {int(ultima['filas_entrenamiento']):,} casos, {calidad} y AUC {ultima['auc']:.3f} "
            f"en la corrida mas reciente ({ultima['fecha']}) - {int(ultima['clientes_calificados']):,} "
            f"clientes calificados."
        )

    if not kpis.empty:
        ultima_fecha = kpis["fecha"].max()
        ultimos = kpis[kpis["fecha"] == ultima_fecha].set_index("kpi")
        alertas = ultimos[ultimos["estado"] == "ALERTA"].index.tolist()

        def _v(kpi: str):
            return ultimos["valor"].get(kpi)

        exposicion = _v("exposicion_total_usd")
        perdida = _v("perdida_esperada_usd")
        if exposicion is not None and perdida is not None:
            puntos.append(
                f"**Exposicion ({ultima_fecha}):** {_formatear_kpi(exposicion, 'USD')} prestados, "
                f"perdida esperada de {_formatear_kpi(perdida, 'USD')} "
                f"({_formatear_kpi(_v('perdida_esperada_pct_exposicion'), '%')} de la cartera); "
                f"{_formatear_kpi(_v('pct_exposicion_riesgo_alto'), '%')} del saldo esta en clientes de riesgo alto."
            )
        if _v("capacidad_ahorro_mediana_pct") is not None:
            puntos.append(
                f"**KPIs de negocio:** capacidad de ahorro de "
                f"{_formatear_kpi(_v('capacidad_ahorro_mediana_pct'), '%')} "
                f"(IC 95%: {_formatear_kpi(_v('capacidad_ahorro_ic95_inf_pct'), '%')} - "
                f"{_formatear_kpi(_v('capacidad_ahorro_ic95_sup_pct'), '%')}), endeudamiento mensual de "
                f"{_formatear_kpi(_v('endeudamiento_mensual_mediana_pct'), '%')} y "
                f"{_formatear_kpi(_v('pct_clientes_semaforo_alto'), '%')} de clientes en semaforo rojo."
            )
        puntos.append(
            f"**Comportamiento:** tasa de ahorro mediana de "
            f"{_formatear_kpi(_v('tasa_ahorro_mediana_pct'), '%')} y "
            f"{_formatear_kpi(_v('pct_clientes_deficitarios'), '%')} de clientes gastan mas de lo que ingresan."
        )
        if alertas:
            puntos.append(
                f"**KPIs en ALERTA ({len(alertas)}):** {', '.join(alertas)} - ver Vista 4."
            )
        else:
            puntos.append("**KPIs financieros:** todos los KPIs con umbral estan dentro de rango.")

    tema.seccion("Resumen de la plataforma", eyebrow="Lectura automática", primera=True,
                 descripcion="Frases armadas con los mismos números de las gráficas: si algo se degrada, el resumen lo dice solo.")
    for punto in puntos:
        st.markdown(f"- {punto}")


# ---------------------------------------------------------------------------
# 7) Vista 5: Modelo de riesgo y dataset de IA (Semanas 14-15)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def _modelo_campeon_cacheado() -> dict:
    return cargar_modelo_campeon()


@st.cache_data(ttl=60)
def _historial_modelos_cacheado() -> pd.DataFrame:
    return cargar_historial_modelos()


def seccion_modelo_ia() -> None:
    tema.seccion("Modelo de riesgo y dataset de IA", eyebrow="Vista 5 · Machine Learning", primera=True)
    st.caption(
        "¿De dónde sale esto? entrenamiento_modelo_dag construye el dataset de features (S14) y entrena, "
        "evalua y registra el modelo (S15) en ml-models/riesgo_credito/. Gold solo carga el campeon."
    )

    registro = _modelo_campeon_cacheado()
    if not registro:
        st.info(
            "Todavía no hay un modelo registrado. Descarga credit_risk con scripts/downloadRawData.py, "
            "corre la malla Bronze/Silver y después entrenamiento_modelo_dag."
        )
        return

    card = registro["card"]
    prueba = card.get("metricas_prueba", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    version = str(card.get("version", ""))
    fecha_version = f"{version[7:9]}/{version[5:7]}/{version[1:5]}" if len(version) >= 9 else "N/D"
    col1.metric("Campeon (entrenado)", fecha_version, help=f"{version} - {registro['campeon'].get('razon')}")
    col2.metric("Modelo", card.get("modelo_elegido", "N/D"), help=card.get("razon_eleccion"))
    col3.metric("AUC (prueba)", f"{prueba.get('auc', float('nan')):.3f}")
    col4.metric("Gini (prueba)", f"{prueba.get('gini', float('nan')):.3f}", help="Gini = 2 x AUC - 1 (estandar de riesgos)")
    col5.metric("KS (prueba)", f"{prueba.get('ks', float('nan')):.3f}", help="Maxima separacion entre buenos y malos")

    # --- Candidatos ---
    st.subheader("Candidatos evaluados")
    filas = []
    for nombre, c in card.get("candidatos", {}).items():
        cv = c["validacion_cruzada"]
        filas.append({
            "candidato": nombre,
            "AUC CV (media ± desv.)": f"{cv['auc']['media']:.3f} ± {cv['auc']['desviacion']:.3f}",
            "Gini CV": f"{cv['gini']['media']:.3f}",
            "AUC prueba": c["prueba"]["auc"],
            "Brier prueba": c["prueba"]["brier"],
            "elegido": "si" if nombre == card.get("modelo_elegido") else "",
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.caption(f"Eleccion: {card.get('razon_eleccion')}. Promocion: {card.get('razon_promocion')}.")

    col_a, col_b = st.columns(2)
    # --- Curva de calibracion: prediccion vs realidad por decil ---
    with col_a:
        st.subheader("Calibración (prueba)")
        curva = pd.DataFrame(card.get("candidatos", {}).get(card.get("modelo_elegido"), {}).get("curva_calibracion", []))
        if not curva.empty:
            fig = px.scatter(
                curva, x="pd_predicha", y="tasa_real", size="clientes",
                labels={"pd_predicha": "PD predicha", "tasa_real": "Impago real"},
            )
            fig.update_traces(marker_color=COLOR_SERIE_PRINCIPAL)
            limite = float(max(curva["pd_predicha"].max(), curva["tasa_real"].max()) * 1.05)
            fig.add_shape(type="line", x0=0, y0=0, x1=limite, y1=limite, line=dict(color=TEXTO_SECUNDARIO, dash="dot"))
            st.plotly_chart(_estilo(fig, mostrar_leyenda=False), use_container_width=True)
            st.caption("Sobre la diagonal = el modelo acierta la tasa de impago de cada grupo.")

    # --- Coeficientes de la scorecard (que pesa cada variable) ---
    with col_b:
        st.subheader("Peso de cada variable")
        coeficientes = card.get("coeficientes_scorecard", {})
        if coeficientes:
            tabla_coef = pd.DataFrame({"variable": list(coeficientes), "coeficiente": list(coeficientes.values())})
            fig_coef = px.bar(
                tabla_coef.sort_values("coeficiente"), x="coeficiente", y="variable", orientation="h",
                labels={"coeficiente": "Coeficiente sobre WoE (scorecard)", "variable": ""},
            )
            fig_coef.update_traces(marker_color=COLOR_SERIE_PRINCIPAL)
            st.plotly_chart(_estilo(fig_coef, mostrar_leyenda=False), use_container_width=True)
            st.caption("Mayor coeficiente = la variable separa mejor a quien paga de quien no.")

    # --- Seleccion de variables (IV) --- a ancho completo y con altura fija:
    # dentro de una columna angosta, Streamlit entra en un ciclo de redibujado.
    st.subheader("Selección de variables (Information Value)")
    fecha_dataset = (card.get("dataset") or {}).get("fecha")
    reporte = cargar_tabla_particion("reporte_seleccion_variables", fecha_dataset) if fecha_dataset else pd.DataFrame()
    if reporte.empty:
        reporte = pd.DataFrame(card.get("features_excluidas", []))
    else:
        reporte = reporte[["variable", "iv", "fuerza", "decision", "razon"]]
    st.dataframe(reporte, hide_index=True, use_container_width=True, height=36 * (len(reporte) + 1) + 4)

    # --- Pruebas de negocio, equidad y deriva ---
    col_c, col_d, col_e = st.columns(3)
    with col_c:
        st.subheader("Pruebas de negocio")
        for variable, resultado in card.get("pruebas_monotonia", {}).items():
            icono = "✓" if resultado["exito"] else "✕"
            st.markdown(f"{icono} **{variable}**: el riesgo {'sube' if resultado['exito'] else 'NO sube'} al empeorar")
    with col_d:
        st.subheader("Equidad (regla 4/5)")
        for nombre, auditoria in (card.get("equidad") or {}).items():
            if auditoria:
                icono = "✓" if auditoria["cumple_regla_4_5"] else "✕"
                st.markdown(f"{icono} **{nombre.replace('_', ' ')}**: ratio de aprobación {auditoria['ratio_aprobacion']:.2f}")
        st.caption("Género y edad NO son variables del modelo; solo se usan para auditar sesgo.")
    with col_e:
        st.subheader("Cambio de distribución (PSI)")
        psi = (card.get("evaluacion_cartera") or {}).get("psi_sin_alinear", {})
        if psi:
            tabla_psi = pd.DataFrame({"variable": list(psi), "PSI sin alinear": list(psi.values())})
            st.dataframe(tabla_psi, hide_index=True, use_container_width=True)
            st.caption("PSI > 0.25 = la cartera es muy distinta al entrenamiento; por eso se alinea por cuantiles.")

    # --- Tabla de puntos e historial ---
    with st.expander("Tabla de puntos de la scorecard"):
        st.dataframe(registro["tabla_puntos"], hide_index=True, use_container_width=True)
    historial = _historial_modelos_cacheado()
    if not historial.empty:
        with st.expander(f"Historial de versiones ({len(historial)})"):
            st.dataframe(historial, hide_index=True, use_container_width=True)
    with st.expander("Limitaciones documentadas"):
        for limitacion in card.get("limitaciones", []):
            st.markdown(f"- {limitacion}")


# ---------------------------------------------------------------------------
# Layout general
# ---------------------------------------------------------------------------
pestanas = st.tabs([
    "Observabilidad", "Resumen", "Salud de la malla", "Calidad de datos",
    "Riesgo de crédito", "KPIs financieros", "Modelo e IA",
])
with pestanas[0]:
    seccion_observabilidad()
with pestanas[1]:
    seccion_resumen_ejecutivo()
with pestanas[2]:
    seccion_salud_malla()
with pestanas[3]:
    seccion_calidad_datos()
with pestanas[4]:
    seccion_riesgo_credito()
with pestanas[5]:
    seccion_kpis_financieros()
with pestanas[6]:
    seccion_modelo_ia()

seccion_chequeo_configuracion()
tema.pie_de_pagina("Gemelo Digital Financiero · Dashboard técnico · Datos de demostración (datasets públicos), "
                   "procesados cada día por Airflow, Spark y MinIO.")
