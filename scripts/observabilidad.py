"""
scripts/observabilidad.py

Semana 18: observabilidad de la plataforma (metricas operativas, SLA y
alertas). Configuracion: config/slas.yml.

Fuentes (todas ya existen; aqui solo se combinan):
  - corridas / tareas: base de metadatos de Airflow, exportada cada hora por
    monitoreo_dag a data/reports/observabilidad/*.parquet
  - eventos: data/reports/pipeline_log.jsonl (lo escriben todos los DAGs;
    incluye las fallas que reporta on_failure_callback)
  - calidad: _calidad_reporte.json de Silver (Great Expectations)
  - modelo: campeon.json + model_card.json del registro de modelos
  - asistente: data/reports/asistente_log.jsonl (una linea por pregunta) y
    data/reports/evaluacion_asistente.json (ultima evaluacion)

Funciones puras sobre DataFrames/dicts: las usan el dashboard tecnico
(Vista 0, siempre fresco) y monitoreo_dag (registra cambios de estado). La
E/S de Airflow vive en el DAG; la de archivos, en cargar_*().
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import yaml

RUTAS_CONFIG = [
    "/opt/airflow/config/slas.yml",
    "/app/slas.yml",  # imagen del dashboard
    os.path.join(os.path.dirname(__file__), "..", "config", "slas.yml"),
]
CARPETA_REPORTES = os.getenv("CARPETA_REPORTES", "/opt/airflow/data/reports")

OK, ALERTA, SIN_DATOS = "OK", "ALERTA", "SIN_DATOS"
ESTADOS_TERMINADOS = {"success", "failed"}


def cargar_config(ruta: str | None = None) -> dict:
    for candidata in ([ruta] if ruta else RUTAS_CONFIG):
        if candidata and os.path.exists(candidata):
            with open(candidata) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No se encontro config/slas.yml en {RUTAS_CONFIG}")


# ---------------------------------------------------------------------------
# Metricas de corridas (Airflow)
# ---------------------------------------------------------------------------
def preparar_corridas(corridas: pd.DataFrame, zona: str) -> pd.DataFrame:
    """Normaliza fechas a la zona de Mexico y calcula duracion y dia de particion."""
    if corridas is None or corridas.empty:
        return pd.DataFrame(columns=["dag_id", "run_id", "run_type", "state", "logical_date",
                                     "start_date", "end_date", "duracion_min", "particion"])
    datos = corridas.copy()
    for columna in ["logical_date", "start_date", "end_date"]:
        datos[columna] = pd.to_datetime(datos[columna], utc=True).dt.tz_convert(zona)
    datos["duracion_min"] = (datos["end_date"] - datos["start_date"]).dt.total_seconds() / 60
    datos["particion"] = datos["logical_date"].dt.normalize()  # dia que procesa (fecha logica)
    return datos


def en_ventana(corridas: pd.DataFrame, ahora: pd.Timestamp, dias: int, columna: str = "start_date") -> pd.DataFrame:
    return corridas[corridas[columna] >= ahora - pd.Timedelta(days=dias)]


def resumen_por_dag(corridas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por DAG: corridas, tasa de exito, duracion tipica y ultimo estado."""
    if corridas.empty:
        return pd.DataFrame()
    filas = []
    for dag_id, grupo in corridas.groupby("dag_id"):
        terminadas = grupo[grupo["state"].isin(ESTADOS_TERMINADOS)]
        exitosas = terminadas[terminadas["state"] == "success"]
        ultima = grupo.sort_values("start_date").iloc[-1]
        filas.append({
            "dag_id": dag_id,
            "corridas": int(len(grupo)),
            "exitosas": int(len(exitosas)),
            "fallidas": int((terminadas["state"] == "failed").sum()),
            "tasa_exito_pct": round(len(exitosas) / len(terminadas) * 100, 1) if len(terminadas) else np.nan,
            "duracion_mediana_min": round(float(exitosas["duracion_min"].median()), 1) if len(exitosas) else np.nan,
            "duracion_p95_min": round(float(exitosas["duracion_min"].quantile(0.95)), 1) if len(exitosas) else np.nan,
            "ultima_corrida": ultima["start_date"],
            "ultimo_estado": ultima["state"],
        })
    return pd.DataFrame(filas).sort_values("dag_id").reset_index(drop=True)


def frescura_por_dia(corridas: pd.DataFrame, dag_id: str, hora_limite: str,
                     ahora: pd.Timestamp, dias: int) -> pd.DataFrame:
    """Para cada dia (particion) de la ventana: ¿Gold quedo listo antes de
    `hora_limite` del dia siguiente? Un dia sin corrida exitosa cuenta como
    tarde (el negocio no tuvo sus datos). Solo cuenta desde el primer dia
    con alguna corrida del DAG, para no castigar fechas previas al proyecto."""
    del_dag = corridas[corridas["dag_id"] == dag_id]
    if del_dag.empty:
        return pd.DataFrame(columns=["particion", "listo_en", "limite", "a_tiempo", "retraso_min"])
    horas, minutos = (int(x) for x in hora_limite.split(":"))
    hoy = ahora.normalize()
    primero = del_dag["particion"].min()
    filas = []
    for dias_atras in range(dias, 0, -1):  # ayer hacia atras; hoy aun no vence
        particion = hoy - pd.Timedelta(days=dias_atras)
        limite = particion + pd.Timedelta(days=1, hours=horas, minutes=minutos)
        if particion < primero or limite > ahora:
            continue
        exitosas = del_dag[(del_dag["particion"] == particion) & (del_dag["state"] == "success")]
        listo = exitosas["end_date"].min() if not exitosas.empty else pd.NaT
        a_tiempo = bool(pd.notna(listo) and listo <= limite)
        retraso = (listo - limite).total_seconds() / 60 if pd.notna(listo) else np.nan
        filas.append({"particion": particion, "listo_en": listo, "limite": limite,
                      "a_tiempo": a_tiempo, "retraso_min": round(retraso, 1) if pd.notna(retraso) else np.nan})
    return pd.DataFrame(filas)


def exito_diario(corridas: pd.DataFrame) -> pd.DataFrame:
    """% de corridas exitosas por dia y DAG (para la tendencia)."""
    terminadas = corridas[corridas["state"].isin(ESTADOS_TERMINADOS)].copy()
    if terminadas.empty:
        return pd.DataFrame(columns=["dia", "dag_id", "corridas", "tasa_exito_pct"])
    terminadas["dia"] = terminadas["start_date"].dt.normalize()
    tabla = terminadas.groupby(["dia", "dag_id"]).agg(
        corridas=("state", "size"), exitosas=("state", lambda s: int((s == "success").sum()))
    ).reset_index()
    tabla["tasa_exito_pct"] = (tabla["exitosas"] / tabla["corridas"] * 100).round(1)
    return tabla


# ---------------------------------------------------------------------------
# Metricas del asistente
# ---------------------------------------------------------------------------
def metricas_asistente(log: pd.DataFrame, ahora: pd.Timestamp, dias: int) -> dict:
    if log is None or log.empty or "timestamp" not in log.columns:
        return {"preguntas": 0}
    datos = log.copy()
    datos["timestamp"] = pd.to_datetime(datos["timestamp"], utc=True, format="ISO8601").dt.tz_convert(ahora.tz)
    datos = datos[datos["timestamp"] >= ahora - pd.Timedelta(days=dias)]
    if datos.empty:
        return {"preguntas": 0}
    modos = datos["modo"].value_counts(normalize=True) * 100
    return {
        "preguntas": int(len(datos)),
        "pct_ia": round(float(modos.get("ia", 0.0)), 1),
        "pct_respaldo": round(float(modos.get("respaldo", 0.0)), 1),
        "pct_basico": round(float(modos.get("basico", 0.0)), 1),
        "pct_con_reintento": round(float((datos.get("reintentos", 0) > 0).mean() * 100), 1),
        "segundos_mediana": round(float(datos["segundos"].median()), 1),
        "segundos_p95": round(float(datos["segundos"].quantile(0.95)), 1),
    }


# ---------------------------------------------------------------------------
# SLA
# ---------------------------------------------------------------------------
def calcular_metricas(entradas: dict, config: dict, ahora: pd.Timestamp) -> dict:
    """Valor actual de cada metrica que puede usar un SLA (None = sin datos)."""
    dias = config["ventana_dias"]
    corridas = entradas.get("corridas", pd.DataFrame())
    monitoreadas = corridas[corridas["dag_id"].isin(config["dags_monitoreados"])] if not corridas.empty else corridas
    recientes = en_ventana(monitoreadas, ahora, dias) if not monitoreadas.empty else monitoreadas
    metricas: dict = {}

    terminadas = recientes[recientes["state"].isin(ESTADOS_TERMINADOS)] if not recientes.empty else recientes
    metricas["tasa_exito_pct"] = (
        round(float((terminadas["state"] == "success").mean() * 100), 1) if len(terminadas) else None
    )

    gold = recientes[(recientes["dag_id"] == "transformacion_gold_dag") & (recientes["state"] == "success")] \
        if not recientes.empty else recientes
    metricas["duracion_gold_p95_min"] = round(float(gold["duracion_min"].quantile(0.95)), 1) if len(gold) else None

    sla_frescura = next((s for s in config["slas"] if s["metrica"] == "frescura_gold_pct"), None)
    frescura = frescura_por_dia(monitoreadas, "transformacion_gold_dag",
                                (sla_frescura or {}).get("hora_limite", "07:00"), ahora, dias) \
        if not monitoreadas.empty else pd.DataFrame()
    metricas["frescura_gold_pct"] = round(float(frescura["a_tiempo"].mean() * 100), 1) if len(frescura) else None

    calidad = entradas.get("calidad", pd.DataFrame())
    if calidad is not None and not calidad.empty and "porcentaje_calidad" in calidad.columns:
        ultimos = calidad.sort_values("fecha").groupby("fuente").tail(1)
        metricas["calidad_silver_min_pct"] = round(float(ultimos["porcentaje_calidad"].min()), 1)
    else:
        metricas["calidad_silver_min_pct"] = None

    modelo = entradas.get("modelo") or {}
    entrenado = modelo.get("fecha_entrenamiento")
    metricas["dias_desde_entrenamiento"] = (
        round((ahora - pd.Timestamp(entrenado).tz_convert(ahora.tz)).total_seconds() / 86400, 1)
        if entrenado else None
    )

    asistente = metricas_asistente(entradas.get("asistente_log"), ahora, dias)
    metricas["asistente_preguntas"] = asistente["preguntas"]
    metricas["asistente_pct_ia"] = asistente.get("pct_ia")
    metricas["asistente_p95_seg"] = asistente.get("segundos_p95")

    evaluacion = entradas.get("evaluacion") or {}
    metricas["evaluacion_exito_final_pct"] = evaluacion.get("exito_final_pct")
    return metricas


def _cumple(valor: float, operador: str, meta: float) -> bool:
    return valor >= meta if operador == ">=" else valor <= meta


def evaluar_slas(entradas: dict, config: dict, ahora: pd.Timestamp) -> pd.DataFrame:
    """Una fila por SLA: valor actual, meta, estado (OK / ALERTA / SIN_DATOS)."""
    metricas = calcular_metricas(entradas, config, ahora)
    filas = []
    for sla in config["slas"]:
        valor = metricas.get(sla["metrica"])
        detalle = ""
        minimo = sla.get("minimo_preguntas")
        if minimo and metricas["asistente_preguntas"] < minimo:
            valor, detalle = None, f"menos de {minimo} preguntas en {config['ventana_dias']} dias"
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            estado = SIN_DATOS
        else:
            estado = OK if _cumple(valor, sla["operador"], sla["meta"]) else ALERTA
        filas.append({
            "sla_id": sla["id"], "nombre": sla["nombre"], "descripcion": sla["descripcion"],
            "valor": valor, "meta": sla["meta"], "operador": sla["operador"], "unidad": sla["unidad"],
            "severidad": sla["severidad"], "estado": estado, "detalle": detalle,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------
def fallas_abiertas(corridas: pd.DataFrame, ahora: pd.Timestamp, horas: int = 48) -> pd.DataFrame:
    """Corridas fallidas recientes que NO tienen una corrida exitosa posterior
    para la misma particion: siguen sin resolverse."""
    if corridas.empty:
        return corridas
    recientes = corridas[corridas["start_date"] >= ahora - pd.Timedelta(hours=horas)]
    fallidas = recientes[recientes["state"] == "failed"]
    abiertas = []
    for _, falla in fallidas.iterrows():
        resuelta = corridas[(corridas["dag_id"] == falla["dag_id"]) & (corridas["particion"] == falla["particion"])
                            & (corridas["state"] == "success") & (corridas["start_date"] > falla["start_date"])]
        if resuelta.empty:
            abiertas.append(falla)
    return pd.DataFrame(abiertas)


def alertas_activas(slas: pd.DataFrame, corridas: pd.DataFrame, eventos: pd.DataFrame,
                    ahora: pd.Timestamp, horas: int = 24) -> pd.DataFrame:
    """Lista unica de alertas, de mas a menos grave."""
    filas = []
    for _, s in slas[slas["estado"] == ALERTA].iterrows():
        filas.append({"severidad": s["severidad"], "origen": "SLA", "elemento": s["nombre"],
                      "mensaje": f"{_fmt(s['valor'], s['unidad'])} (meta {s['operador']} {_fmt(s['meta'], s['unidad'])})",
                      "desde": pd.NaT})
    for _, f in fallas_abiertas(corridas, ahora).iterrows():
        filas.append({"severidad": "alta", "origen": "Corrida fallida", "elemento": f["dag_id"],
                      "mensaje": f"Particion {f['particion']:%Y-%m-%d} sin corrida exitosa posterior",
                      "desde": f["start_date"]})
    if eventos is not None and not eventos.empty and {"nivel", "timestamp"} <= set(eventos.columns):
        datos = eventos.copy()
        datos["timestamp"] = pd.to_datetime(datos["timestamp"], utc=True, format="ISO8601").dt.tz_convert(ahora.tz)
        fallas = datos[(datos["timestamp"] >= ahora - pd.Timedelta(hours=horas))
                       & (datos.get("tipo", pd.Series(index=datos.index, dtype=object)) == "falla_tarea")]
        for _, e in fallas.iterrows():
            filas.append({"severidad": "alta", "origen": "Tarea fallida",
                          "elemento": f"{e.get('dag_id')} · {e.get('task_id')}",
                          "mensaje": str(e.get("error", ""))[:160], "desde": e["timestamp"]})
    if not filas:
        return pd.DataFrame(columns=["severidad", "origen", "elemento", "mensaje", "desde"])
    tabla = pd.DataFrame(filas)
    tabla["_orden"] = tabla["severidad"].map({"alta": 0, "media": 1}).fillna(2)
    return tabla.sort_values(["_orden", "desde"], ascending=[True, False]).drop(columns="_orden").reset_index(drop=True)


def estado_general(slas: pd.DataFrame, alertas: pd.DataFrame) -> str:
    """Critico si hay alguna alerta alta; Atencion si hay alguna media; si no, Saludable."""
    if not alertas.empty and (alertas["severidad"] == "alta").any():
        return "Crítico"
    if not alertas.empty:
        return "Requiere atención"
    if not slas.empty and (slas["estado"] == SIN_DATOS).all():
        return "Sin datos"
    return "Saludable"


def _fmt(valor, unidad: str) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "N/D"
    texto = f"{valor:,.1f}".rstrip("0").rstrip(".")
    return f"{texto}%" if unidad == "%" else f"{texto} {unidad}"


# ---------------------------------------------------------------------------
# E/S de archivos (las comparten el DAG y el dashboard)
# ---------------------------------------------------------------------------
def carpeta(base: str | None = None) -> str:
    return os.path.join(base or CARPETA_REPORTES, "observabilidad")


def cargar_jsonl(ruta: str) -> pd.DataFrame:
    if not os.path.exists(ruta):
        return pd.DataFrame()
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError:
                continue  # una linea a medio escribir no tumba el tablero
    return pd.DataFrame(filas)


def cargar_entradas_locales(base: str | None = None) -> dict:
    """Lo que esta en data/reports: corridas/tareas exportadas, eventos,
    log y evaluacion del asistente. (Calidad y modelo vienen de MinIO.)"""
    base = base or CARPETA_REPORTES
    entradas = {}
    for nombre in ["corridas", "tareas"]:
        ruta = os.path.join(carpeta(base), f"{nombre}.parquet")
        entradas[nombre] = pd.read_parquet(ruta) if os.path.exists(ruta) else pd.DataFrame()
    entradas["eventos"] = cargar_jsonl(os.path.join(base, "pipeline_log.jsonl"))
    entradas["asistente_log"] = cargar_jsonl(os.path.join(base, "asistente_log.jsonl"))
    ruta_eval = os.path.join(base, "evaluacion_asistente.json")
    entradas["evaluacion"] = json.load(open(ruta_eval)) if os.path.exists(ruta_eval) else {}
    ruta_meta = os.path.join(carpeta(base), "exportacion.json")
    entradas["exportacion"] = json.load(open(ruta_meta)) if os.path.exists(ruta_meta) else {}
    return entradas


def registrar_pregunta(registro: dict, base: str | None = None) -> None:
    """Una linea por pregunta al asistente (la escribe el dashboard ejecutivo).
    Nunca debe romper el tablero: cualquier error de disco se ignora."""
    try:
        ruta = os.path.join(base or CARPETA_REPORTES, "asistente_log.jsonl")
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def cargar_calidad_minio(fs, bucket: str) -> pd.DataFrame:
    """Ultimos reportes de Great Expectations de Silver (uno por fuente y fecha)."""
    filas = []
    try:
        rutas = [r for r in fs.find(bucket) if r.endswith("_calidad_reporte.json")]
    except FileNotFoundError:
        return pd.DataFrame()
    for ruta in rutas:
        try:
            with fs.open(ruta) as f:
                reporte = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        totales = reporte.get("expectativas_totales") or 0
        filas.append({"fuente": reporte.get("fuente"), "fecha": reporte.get("fecha"),
                      "porcentaje_calidad": round(reporte.get("expectativas_exitosas", 0) / totales * 100, 1)
                      if totales else None})
    return pd.DataFrame(filas)


def cargar_modelo_minio(fs, bucket: str) -> dict:
    """Fecha del ultimo entrenamiento (cualquier version, promovida o no) y
    version campeona. Entrenar y conservar al campeon tambien es "estar al dia"."""
    raiz = f"{bucket}/riesgo_credito"
    try:
        tarjetas = [r for r in fs.find(raiz) if r.endswith("model_card.json")]
    except FileNotFoundError:
        return {}
    fechas = []
    for ruta in tarjetas:
        try:
            with fs.open(ruta) as f:
                fechas.append(json.load(f).get("entrenado_en"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    fechas = [f for f in fechas if f]
    try:
        with fs.open(f"{raiz}/campeon.json") as f:
            campeon = json.load(f).get("version")
    except FileNotFoundError:
        campeon = None
    return {"fecha_entrenamiento": max(fechas) if fechas else None, "version_campeon": campeon}
