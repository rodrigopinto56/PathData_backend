"""
scripts/reporte_ejecutivo.py

Semana 13: capa de "lenguaje de negocio" para el dashboard ejecutivo.

Traduce las tablas de KPIs de Gold (gold_kpis_financieros y
gold_kpis_por_segmento) a lo que necesita alguien que NO es programador:
  - un diagnostico en una frase (Saludable / Requiere atencion / Critico),
  - los 4 KPIs de negocio explicados con palabras ("de cada $100..."),
  - el semaforo de clientes y el "dinero en juego",
  - acciones sugeridas generadas a partir de los KPIs en alerta,
  - un reporte HTML de una pagina, imprimible, que el DAG publica cada dia.

Lo usan dos consumidores, por eso vive aqui y no dentro del dashboard:
  - dashboard/ejecutivo.py (Streamlit), en vivo.
  - transformacion_gold_dag -> tarea publicar_reporte_ejecutivo.

Todo es pandas puro excepto publicar_reporte_particion() (E/S a MinIO).
"""

from __future__ import annotations

import html
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

GOLD_BUCKET = os.getenv("GOLD_BUCKET", "gold-layer")
CARPETA_REPORTES_LOCAL = "/opt/airflow/data/reports/ejecutivo"

# --- Estados en lenguaje de negocio ---
SALUDABLE = "Saludable"
ATENCION = "Requiere atención"
CRITICO = "Crítico"
SIN_DATOS = "Sin datos"
ESTADO_NEGOCIO = {"OK": SALUDABLE, "ALERTA": ATENCION, "SIN_DATOS": SIN_DATOS, "INFO": "Informativo"}

# Paleta de estado (fija): siempre acompanada de icono + texto, nunca solo color.
COLOR_ESTADO = {SALUDABLE: "#0ca30c", ATENCION: "#fab219", CRITICO: "#d03b3b", SIN_DATOS: "#8a8984"}
ICONO_ESTADO = {SALUDABLE: "✓", ATENCION: "!", CRITICO: "✕", SIN_DATOS: "–"}

# Semaforo de clientes (indice de riesgo consolidado)
NIVELES_SEMAFORO = [
    ("Bajo", "Riesgo bajo", COLOR_ESTADO[SALUDABLE]),
    ("Medio", "Riesgo medio", COLOR_ESTADO[ATENCION]),
    ("Alto", "Riesgo alto", COLOR_ESTADO[CRITICO]),
]

# --- Nombres "humanos" para dimensiones y valores del dataset ---
NOMBRE_DIMENSION = {
    "region": "Región",
    "rango_credit_score": "Calificación crediticia",
    "tipo_prestamo": "Tipo de préstamo",
    "rango_edad": "Edad",
    "employment_status": "Situación laboral",
    "perfil_ahorro": "Perfil de ahorro",
}
TRADUCCION_VALORES = {
    "Employed": "Empleado",
    "Self-employed": "Independiente",
    "Unemployed": "Desempleado",
    "Student": "Estudiante",
    "Retired": "Jubilado",
    "Sin prestamo": "Sin préstamo",
    "Home": "Hipotecario",
    "Car": "Automotriz",
    "Personal": "Personal",
    "Education": "Educativo",
    "Business": "Negocio",
    "Medical": "Médico",
    "North": "Norte",
    "South": "Sur",
    "East": "Este",
    "West": "Oeste",
    "Central": "Centro",
}

MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# Nombre de negocio de los KPIs que cuentan como "indicadores clave"
INDICADORES_CLAVE = {
    "capacidad_ahorro_mediana_pct": "capacidad de ahorro",
    "endeudamiento_mensual_mediana_pct": "endeudamiento mensual",
    "indice_riesgo_consolidado_promedio": "índice de riesgo",
    "pct_clientes_semaforo_alto": "clientes en rojo",
}

# Cuantos puntos por encima del promedio debe estar un segmento para
# senalarlo como "foco de riesgo" (evita senalar diferencias de ruido).
MARGEN_FOCO_RIESGO = 2.0


# ---------------------------------------------------------------------------
# Formato de numeros (como los leeria un ejecutivo)
# ---------------------------------------------------------------------------
def _hay(valor) -> bool:
    return valor is not None and not pd.isna(valor)


def fmt_usd(valor) -> str:
    if not _hay(valor):
        return "N/D"
    if abs(valor) >= 1_000_000_000:
        return f"${valor / 1_000_000_000:,.1f} mil millones"
    if abs(valor) >= 1_000_000:
        return f"${valor / 1_000_000:,.1f} millones"
    return f"${valor:,.0f}"


def fmt_pct(valor, decimales: int = 1) -> str:
    return f"{valor:.{decimales}f}%" if _hay(valor) else "N/D"


def fmt_int(valor) -> str:
    return f"{int(round(valor)):,}" if _hay(valor) else "N/D"


def traducir(valor: str) -> str:
    return TRADUCCION_VALORES.get(str(valor), str(valor))


def fecha_legible(fecha: str) -> str:
    """'2026/09/22' -> '22 sep 2026'."""
    try:
        anio, mes, dia = str(fecha).split("/")
        return f"{int(dia)} {MESES[int(mes) - 1]} {anio}"
    except (ValueError, IndexError):
        return str(fecha)


# ---------------------------------------------------------------------------
# Resumen ejecutivo (estructura de datos que pintan dashboard y reporte)
# ---------------------------------------------------------------------------
def _valores(kpis: pd.DataFrame) -> tuple[dict, dict]:
    tabla = kpis.drop_duplicates("kpi", keep="last").set_index("kpi")
    valores = {k: (None if pd.isna(v) else float(v)) for k, v in tabla["valor"].items()}
    estados = tabla["estado"].to_dict()
    return valores, estados


def _total_clientes(segmentos: pd.DataFrame) -> float | None:
    if segmentos is None or segmentos.empty:
        return None
    por_region = segmentos[segmentos["dimension"] == "region"]
    return float(por_region["num_clientes"].sum()) if not por_region.empty else None


def _tarjetas(v: dict, e: dict) -> list[dict]:
    ahorro = v.get("capacidad_ahorro_mediana_pct")
    endeudamiento = v.get("endeudamiento_mensual_mediana_pct")
    indice = v.get("indice_riesgo_consolidado_promedio")

    return [
        {
            "titulo": "Capacidad de ahorro",
            "valor": fmt_pct(ahorro),
            "explicacion": (
                f"De cada $100 que gana el cliente típico, le quedan ${ahorro:.0f} después de sus gastos."
                if _hay(ahorro) else "No se pudo calcular."
            ),
            "meta": "Meta: 20% o más",
            "nota": (
                f"Rango confiable: {fmt_pct(v.get('capacidad_ahorro_ic95_inf_pct'))} a "
                f"{fmt_pct(v.get('capacidad_ahorro_ic95_sup_pct'))}"
                if _hay(v.get("capacidad_ahorro_ic95_inf_pct")) else ""
            ),
            "estado": ESTADO_NEGOCIO.get(e.get("capacidad_ahorro_mediana_pct"), SIN_DATOS),
        },
        {
            "titulo": "Endeudamiento mensual",
            "valor": fmt_pct(endeudamiento),
            "explicacion": (
                f"De cada $100 de ingreso, ${endeudamiento:.0f} se van al pago de deudas (clientes con préstamo)."
                if _hay(endeudamiento) else "Faltan datos del pago mensual de los préstamos."
            ),
            "meta": "Meta: menos de 36%",
            "nota": (
                f"{fmt_pct(v.get('pct_clientes_endeudamiento_alto'))} de los clientes con préstamo supera la meta"
                if _hay(v.get("pct_clientes_endeudamiento_alto")) else ""
            ),
            "estado": ESTADO_NEGOCIO.get(e.get("endeudamiento_mensual_mediana_pct"), SIN_DATOS),
        },
        {
            "titulo": "Índice de riesgo",
            "valor": f"{indice:.0f} / 100" if _hay(indice) else "N/D",
            "explicacion": "Calificación que combina endeudamiento, probabilidad de impago, ahorro e historial crediticio. 0 = muy sano, 100 = muy riesgoso.",
            "meta": "Meta: menos de 40",
            "nota": "",
            "estado": ESTADO_NEGOCIO.get(e.get("indice_riesgo_consolidado_promedio"), SIN_DATOS),
        },
        {
            "titulo": "Estabilidad de ingresos y gastos",
            "valor": "Pendiente",
            "explicacion": "Mide si los ingresos y gastos del cliente cambian mucho mes a mes. Requiere de 3 a 6 meses de historial por cliente, que hoy no tenemos.",
            "meta": "Meta: tendencia estable",
            "nota": "Se incorporará en una siguiente fase",
            "estado": ESTADO_NEGOCIO.get(e.get("varianza_ingresos_gastos"), SIN_DATOS),
        },
    ]


def _diagnostico(v: dict, e: dict, total: float | None) -> dict:
    pct_bajo = v.get("pct_clientes_semaforo_bajo")
    pct_medio = v.get("pct_clientes_semaforo_medio")
    pct_alto = v.get("pct_clientes_semaforo_alto")
    indice = v.get("indice_riesgo_consolidado_promedio")

    en_alerta = [nombre for kpi, nombre in INDICADORES_CLAVE.items() if e.get(kpi) == "ALERTA"]

    if (_hay(pct_alto) and pct_alto > 30) or (_hay(indice) and indice >= 55):
        nivel = CRITICO
    elif en_alerta:
        nivel = ATENCION
    else:
        nivel = SALUDABLE

    de_cuantos = f"De {fmt_int(total)} clientes, " if _hay(total) else "De los clientes, "
    frase = (
        f"{de_cuantos}{fmt_pct(pct_bajo, 0)} está en verde, {fmt_pct(pct_medio, 0)} en amarillo "
        f"y {fmt_pct(pct_alto, 0)} en rojo."
    )
    if en_alerta:
        frase += f" Fuera de meta: {', '.join(en_alerta)}."
    else:
        frase += " Todos los indicadores clave están dentro de meta."
    return {"nivel": nivel, "frase": frase}


def _foco_riesgo(segmentos: pd.DataFrame, dimension: str = "region") -> dict | None:
    """El segmento con mayor indice de riesgo, si sobresale del resto."""
    if segmentos is None or segmentos.empty or "indice_riesgo_promedio" not in segmentos.columns:
        return None
    datos = segmentos[segmentos["dimension"] == dimension].dropna(subset=["indice_riesgo_promedio"])
    if len(datos) < 2:
        return None
    promedio = (datos["indice_riesgo_promedio"] * datos["num_clientes"]).sum() / datos["num_clientes"].sum()
    peor = datos.sort_values("indice_riesgo_promedio", ascending=False).iloc[0]
    if peor["indice_riesgo_promedio"] - promedio < MARGEN_FOCO_RIESGO:
        return None
    return {
        "segmento": traducir(peor["segmento"]),
        "indice": float(peor["indice_riesgo_promedio"]),
        "promedio": float(promedio),
    }


def _acciones(v: dict, e: dict, total: float | None, segmentos: pd.DataFrame) -> list[dict]:
    """Reglas: cada KPI en alerta se traduce en una accion concreta.
    Se ordenan por prioridad (Alta primero)."""
    acciones = []

    def agregar(prioridad, titulo, detalle):
        acciones.append({"prioridad": prioridad, "titulo": titulo, "detalle": detalle})

    pct_alto = v.get("pct_clientes_semaforo_alto")
    if e.get("pct_clientes_semaforo_alto") == "ALERTA" or (_hay(pct_alto) and pct_alto > 0 and _hay(total)):
        n_rojos = fmt_int(pct_alto / 100 * total) if _hay(pct_alto) and _hay(total) else "Los"
        agregar(
            "Alta" if e.get("pct_clientes_semaforo_alto") == "ALERTA" else "Media",
            f"Contactar a los {n_rojos} clientes en rojo",
            "Revisar su situación antes de que caigan en impago: reestructura, ajuste de plazo o asesoría financiera. "
            "La lista está en la sección «Clientes prioritarios».",
        )

    if e.get("endeudamiento_mensual_mediana_pct") == "ALERTA" or e.get("pct_clientes_endeudamiento_alto") == "ALERTA":
        agregar(
            "Alta",
            "Ofrecer consolidación o reestructura de deuda",
            f"{fmt_pct(v.get('pct_clientes_endeudamiento_alto'))} de los clientes con préstamo destina más del 36% "
            "de su ingreso a deudas; bajar la mensualidad reduce su probabilidad de impago.",
        )

    if e.get("pct_exposicion_riesgo_alto") == "ALERTA":
        agregar(
            "Alta",
            "Revisar la política de montos de crédito",
            f"{fmt_pct(v.get('pct_exposicion_riesgo_alto'))} del dinero prestado está en clientes de riesgo alto. "
            "Conviene limitar montos o pedir garantías en ese perfil.",
        )

    if e.get("perdida_esperada_pct_exposicion") == "ALERTA":
        agregar(
            "Media",
            "Revisar el nivel de reservas",
            f"Por cada $100 prestados se espera perder ${v.get('perdida_esperada_pct_exposicion', 0):.0f} "
            f"(pérdida esperada total: {fmt_usd(v.get('perdida_esperada_usd'))}).",
        )

    if e.get("capacidad_ahorro_mediana_pct") == "ALERTA":
        agregar(
            "Media",
            "Impulsar productos de ahorro automático",
            "El cliente típico ahorra menos del 20% de su ingreso; un ahorro programado mejora su capacidad para enfrentar imprevistos.",
        )

    if e.get("pct_clientes_sin_colchon") == "ALERTA" or e.get("meses_cobertura_ahorro_mediana") == "ALERTA":
        agregar(
            "Media",
            "Promover un fondo de emergencia",
            f"{fmt_pct(v.get('pct_clientes_sin_colchon'))} de los clientes no podría cubrir 3 meses de gastos con su ahorro.",
        )

    if e.get("pct_clientes_deficitarios") == "ALERTA":
        agregar(
            "Media",
            "Acompañar a clientes que gastan más de lo que ganan",
            f"{fmt_pct(v.get('pct_clientes_deficitarios'))} de los clientes tiene gastos mayores a su ingreso.",
        )

    if e.get("indice_concentracion_region_hhi") == "ALERTA":
        agregar(
            "Media",
            "Diversificar la colocación por región",
            f"{fmt_pct(v.get('pct_exposicion_region_principal'))} del dinero prestado está en una sola región.",
        )

    foco = _foco_riesgo(segmentos)
    if foco:
        agregar(
            "Media",
            f"Priorizar la región {foco['segmento']}",
            f"Tiene el mayor índice de riesgo ({foco['indice']:.0f} vs {foco['promedio']:.0f} de promedio).",
        )

    if not acciones:
        agregar("Seguimiento", "Mantener el monitoreo mensual", "Todos los indicadores clave están dentro de meta.")

    orden = {"Alta": 0, "Media": 1, "Seguimiento": 2}
    return sorted(acciones, key=lambda a: orden[a["prioridad"]])


def resumen_ejecutivo(kpis: pd.DataFrame, segmentos: pd.DataFrame | None = None) -> dict | None:
    """Arma todo lo que muestra el dashboard ejecutivo para UNA fecha.

    Regresa None si la particion no tiene los KPIs de negocio (se calculo
    con una version anterior del pipeline): hay que re-ejecutar el DAG Gold."""
    if kpis is None or kpis.empty or "indice_riesgo_consolidado_promedio" not in set(kpis["kpi"]):
        return None
    segmentos = segmentos if segmentos is not None else pd.DataFrame()
    v, e = _valores(kpis)
    total = _total_clientes(segmentos)

    return {
        "fecha": str(kpis["fecha"].iloc[0]),
        "total_clientes": total,
        "diagnostico": _diagnostico(v, e, total),
        "tarjetas": _tarjetas(v, e),
        "semaforo": [
            {
                "nivel": nivel,
                "etiqueta": etiqueta,
                "color": color,
                "pct": v.get(f"pct_clientes_semaforo_{nivel.lower()}"),
                "clientes": (
                    v.get(f"pct_clientes_semaforo_{nivel.lower()}") / 100 * total
                    if _hay(v.get(f"pct_clientes_semaforo_{nivel.lower()}")) and _hay(total) else None
                ),
            }
            for nivel, etiqueta, color in NIVELES_SEMAFORO
        ],
        "dinero": {
            "exposicion": v.get("exposicion_total_usd"),
            "num_prestamos": v.get("num_prestamos_activos"),
            "perdida_esperada": v.get("perdida_esperada_usd"),
            "perdida_por_cada_100": v.get("perdida_esperada_pct_exposicion"),
            "pct_en_riesgo_alto": v.get("pct_exposicion_riesgo_alto"),
        },
        "acciones": _acciones(v, e, total, segmentos),
    }


def riesgo_por_segmento(segmentos: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Tabla "¿donde esta el riesgo?" con nombres en espanol, ordenada de
    mayor a menor indice de riesgo."""
    if segmentos is None or segmentos.empty:
        return pd.DataFrame()
    datos = segmentos[segmentos["dimension"] == dimension].copy()
    if datos.empty:
        return datos
    datos["Segmento"] = datos["segmento"].map(traducir)
    columnas = {
        "num_clientes": "Clientes",
        "indice_riesgo_promedio": "Índice de riesgo",
        "pct_semaforo_alto": "% en rojo",
        "exposicion_usd": "Dinero prestado",
        "perdida_esperada_usd": "Pérdida esperada",
        "tasa_ahorro_mediana_pct": "Ahorro típico %",
    }
    disponibles = {k: n for k, n in columnas.items() if k in datos.columns}
    tabla = datos[["Segmento", *disponibles]].rename(columns=disponibles)
    orden = "Índice de riesgo" if "Índice de riesgo" in tabla.columns else "Clientes"
    return tabla.sort_values(orden, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporte HTML de una pagina (se publica cada dia y se descarga del dashboard)
# ---------------------------------------------------------------------------
def _e(texto) -> str:
    return html.escape(str(texto))


def _chip(estado: str) -> str:
    color = COLOR_ESTADO.get(estado, COLOR_ESTADO[SIN_DATOS])
    return (
        f'<span class="chip" style="border-color:{color}">'
        f'<span class="dot" style="background:{color}">{ICONO_ESTADO.get(estado, "–")}</span>{_e(estado)}</span>'
    )


def generar_reporte_html(resumen: dict, segmentos: pd.DataFrame | None = None) -> str:
    """HTML autocontenido (sin JavaScript ni recursos externos): se puede
    mandar por correo, abrir en cualquier navegador e imprimir a PDF."""
    tarjetas = "".join(
        f"""<div class="card">
              {_chip(t['estado'])}<h3 style="margin-top:8px">{_e(t['titulo'])}</h3>
              <div class="big">{_e(t['valor'])}</div>
              <p>{_e(t['explicacion'])}</p>
              <p class="muted">{_e(t['meta'])}{' · ' + _e(t['nota']) if t['nota'] else ''}</p>
            </div>"""
        for t in resumen["tarjetas"]
    )

    barra = "".join(
        f'<div class="seg" style="width:{s["pct"] or 0}%;background:{s["color"]}" '
        f'title="{_e(s["etiqueta"])}: {fmt_pct(s["pct"])}"></div>'
        for s in resumen["semaforo"]
    )
    leyenda = "".join(
        f'<li><span class="sq" style="background:{s["color"]}"></span>{_e(s["etiqueta"])}: '
        f'<b>{fmt_pct(s["pct"])}</b> ({fmt_int(s["clientes"])} clientes)</li>'
        for s in resumen["semaforo"]
    )

    acciones = "".join(
        f'<li><span class="prio prio-{_e(a["prioridad"].lower())}">{_e(a["prioridad"])}</span>'
        f'<b>{_e(a["titulo"])}.</b> {_e(a["detalle"])}</li>'
        for a in resumen["acciones"]
    )

    regiones = riesgo_por_segmento(segmentos, "region") if segmentos is not None else pd.DataFrame()
    filas_region = ""
    if not regiones.empty:
        for _, fila in regiones.iterrows():
            filas_region += (
                f"<tr><td>{_e(fila['Segmento'])}</td><td>{fmt_int(fila['Clientes'])}</td>"
                f"<td>{fila.get('Índice de riesgo', float('nan')):.0f}</td>"
                f"<td>{fmt_pct(fila.get('% en rojo'))}</td>"
                f"<td>{fmt_usd(fila.get('Dinero prestado'))}</td></tr>"
            )
    tabla_region = (
        f"""<h2>¿Dónde está el riesgo? — por región</h2>
            <table><thead><tr><th>Región</th><th>Clientes</th><th>Índice de riesgo</th>
            <th>% en rojo</th><th>Dinero prestado</th></tr></thead><tbody>{filas_region}</tbody></table>"""
        if filas_region else ""
    )

    d = resumen["dinero"]
    diag = resumen["diagnostico"]
    color_diag = COLOR_ESTADO[diag["nivel"]]
    generado = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte ejecutivo {_e(fecha_legible(resumen['fecha']))}</title>
<style>
  /* Sin fuentes externas (el reporte se manda por correo y debe verse
     igual sin internet): usa Source Serif 4 / IBM Plex Sans si estan
     instaladas y si no Georgia / Segoe UI, presentes en casi todo equipo. */
  :root {{ --serif:'Source Serif 4',Georgia,'Times New Roman',serif; --sans:'IBM Plex Sans','Segoe UI','Helvetica Neue',Arial,sans-serif;
           --surface:#fcfcfb; --card:#ffffff; --ink:#0b0b0b; --ink-2:#52514e; --line:#e4e3df; --accent:#072146; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--surface); color:var(--ink); font:15px/1.55 var(--sans); }}
  main {{ max-width:960px; margin:0 auto; padding:32px 16px; }}
  header {{ border-bottom:3px solid var(--accent); padding-bottom:12px; margin-bottom:20px; }}
  h1 {{ margin:0; font-family:var(--serif); font-size:30px; font-weight:700; letter-spacing:-0.01em; color:var(--accent); }}
  h2 {{ font-family:var(--serif); font-size:20px; font-weight:600; margin:30px 0 10px; color:var(--accent); }}
  h3 {{ margin:0; font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-2); }}
  .muted {{ color:var(--ink-2); font-size:13px; }}
  .diag {{ border-left:6px solid {color_diag}; background:var(--card); padding:14px 16px; border-radius:8px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; }}
  .card-top {{ display:flex; justify-content:space-between; gap:8px; align-items:flex-start; }}
  .card p {{ margin:6px 0 0; font-size:13px; }}
  .big {{ font-size:30px; font-weight:600; margin-top:8px; letter-spacing:-0.02em; font-variant-numeric:tabular-nums lining-nums; }}
  .chip {{ display:inline-flex; align-items:center; gap:5px; border:1.5px solid; border-radius:999px; padding:1px 8px 1px 2px; font-size:12px; white-space:nowrap; }}
  .dot {{ display:inline-grid; place-items:center; width:16px; height:16px; border-radius:50%; color:#fff; font-size:11px; font-weight:700; }}
  .bar {{ display:flex; height:22px; border-radius:6px; overflow:hidden; gap:2px; background:var(--surface); }}
  .seg {{ height:100%; }}
  ul.leg {{ list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:6px 20px; }}
  .sq {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }}
  .money {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; }}
  ol.acc {{ padding-left:20px; }} ol.acc li {{ margin:8px 0; }}
  .prio {{ display:inline-block; font-size:11px; font-weight:700; border-radius:4px; padding:0 6px; margin-right:8px; border:1px solid var(--ink-2); }}
  .prio-alta {{ border-color:#d03b3b; color:#d03b3b; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); font-size:14px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--ink-2); font-weight:600; }}
  footer {{ margin-top:28px; border-top:1px solid var(--line); padding-top:10px; }}
  @media print {{ body {{ background:#fff; }} main {{ padding:0; }} .card, .diag {{ break-inside:avoid; }} }}
</style></head>
<body><main>
<header>
  <h1>Salud financiera de la cartera</h1>
  <div class="muted">Gemelo Digital Financiero · Corte al {_e(fecha_legible(resumen['fecha']))} · {fmt_int(resumen['total_clientes'])} clientes</div>
</header>

<div class="diag">{_chip(diag['nivel'])}<p style="margin:8px 0 0">{_e(diag['frase'])}</p></div>

<h2>Indicadores clave</h2>
<div class="grid">{tarjetas}</div>

<h2>¿Cómo están nuestros clientes?</h2>
<div class="bar" role="img" aria-label="Distribución de clientes por nivel de riesgo">{barra}</div>
<ul class="leg">{leyenda}</ul>

<h2>Dinero en juego</h2>
<div class="money">
  <div class="card"><h3>Dinero prestado</h3><div class="big">{fmt_usd(d['exposicion'])}</div>
    <p class="muted">{fmt_int(d['num_prestamos'])} préstamos activos</p></div>
  <div class="card"><h3>Pérdida esperada</h3><div class="big">{fmt_usd(d['perdida_esperada'])}</div>
    <p class="muted">Por cada $100 prestados se esperan perder ${(d['perdida_por_cada_100'] or 0):.0f}</p></div>
  <div class="card"><h3>Prestado a clientes de riesgo alto</h3><div class="big">{fmt_pct(d['pct_en_riesgo_alto'])}</div>
    <p class="muted">del total prestado</p></div>
</div>

<h2>Acciones sugeridas</h2>
<ol class="acc">{acciones}</ol>

{tabla_region}

<footer class="muted">
  Generado automáticamente el {generado} por el pipeline de datos (transformacion_gold_dag).
  Datos de demostración (dataset sintético público). La estabilidad de ingresos y gastos requiere historial mensual y se incorporará en una siguiente fase.
</footer>
</main></body></html>
"""


# ---------------------------------------------------------------------------
# E/S: publicar el reporte de una particion (lo llama el DAG)
# ---------------------------------------------------------------------------
def publicar_reporte_particion(fecha: str) -> dict:
    """Lee los KPIs de `fecha` de gold-layer, genera el reporte HTML y lo
    publica en MinIO (gold-layer/reportes_ejecutivos/) y en disco local
    (data/reports/ejecutivo/) para poder compartirlo."""
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET, STORAGE_OPTIONS

    kpis = pd.read_parquet(
        f"s3://{GOLD_BUCKET}/gold_kpis_financieros/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS
    )
    segmentos = pd.read_parquet(
        f"s3://{GOLD_BUCKET}/gold_kpis_por_segmento/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS
    )

    resumen = resumen_ejecutivo(kpis, segmentos)
    if resumen is None:
        raise ValueError(f"La particion {fecha} no tiene KPIs de negocio; re-ejecuta calcular_kpis.")
    contenido = generar_reporte_html(resumen, segmentos)

    nombre = f"reporte_ejecutivo_{fecha.replace('/', '-')}.html"
    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    ruta_minio = f"{GOLD_BUCKET}/reportes_ejecutivos/{nombre}"
    with fs.open(ruta_minio, "w") as f:
        f.write(contenido)

    ruta_local = None
    if os.path.isdir(os.path.dirname(CARPETA_REPORTES_LOCAL)):
        os.makedirs(CARPETA_REPORTES_LOCAL, exist_ok=True)
        ruta_local = f"{CARPETA_REPORTES_LOCAL}/{nombre}"
        with open(ruta_local, "w", encoding="utf-8") as f:
            f.write(contenido)

    return {
        "fecha": fecha,
        "diagnostico": resumen["diagnostico"]["nivel"],
        "acciones_sugeridas": len(resumen["acciones"]),
        "ruta_minio": f"s3://{ruta_minio}",
        "ruta_local": ruta_local,
    }


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(publicar_reporte_particion(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
