"""
scripts/evaluar_asistente.py

Semana 17: evaluacion automatica del asistente financiero.

Banco de 50 preguntas (S19; antes 30) con respuesta CONOCIDA (las cifras correctas se calculan
de las mismas tablas Gold). Para cada pregunta mide:

  - Enrutamiento: ¿consulto los datos correctos?
  - Veracidad:    ¿todas las cifras de la respuesta existen en los datos?
  - Exito:        ¿la respuesta contiene el dato que se pregunto (y no se nego)?

Se reporta para la respuesta "cruda" del LLM (primer intento) y para la
respuesta FINAL que ve el usuario (despues del control de calidad, el
reintento y el respaldo). Meta del proyecto: >= 84%.

Uso (dentro del contenedor del dashboard ejecutivo, que tiene acceso a
MinIO y a Ollama):
    docker compose exec streamlit-ejecutivo python evaluar_asistente.py
    docker compose exec streamlit-ejecutivo python evaluar_asistente.py --fecha 2026/10/28 --modelo llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

import asistente_financiero as asis
import reporte_ejecutivo as rep

META_PCT = 84.0
# En el contenedor del dashboard ejecutivo: /app/data/reports (volumen ./data)
CARPETA_REPORTES = os.getenv("CARPETA_REPORTES", ".")


@dataclass
class Caso:
    pregunta: str
    consulta: str                                      # consulta que DEBE elegir el enrutador
    hechos: Callable[[asis.DatosAsistente], list[str]]  # respuestas aceptables (basta una)


def _kpi(datos, kpi):
    fila = datos.kpis[datos.kpis["kpi"] == kpi]
    return None if fila.empty else fila.iloc[0]


def _fmt(datos, kpi) -> list[str]:
    fila = _kpi(datos, kpi)
    return [] if fila is None else [asis._fmt_kpi(fila["valor"], fila.get("unidad"))]


def _estado(datos, kpi) -> list[str]:
    """Frases que expresan correctamente si el KPI cumple su meta.
    S19: se acepta "se cumple / no se cumple" (antes solo "cumple / no cumple",
    y "Sí, se cumple" contaba como error aunque fuera correcto)."""
    fila = _kpi(datos, kpi)
    if fila is None:
        return []
    return {"OK": ["sí se cumple", "sí, se cumple", "sí cumple", "Saludable"],
            "ALERTA": ["no se cumple", "no, no se cumple", "no cumple", "Requiere atención"]}.get(fila["estado"], [])


def _top_segmento(datos, dimension) -> list[str]:
    """El segmento de mayor riesgo; si hay empate (menos de 1 punto), cualquiera de los empatados."""
    tabla = rep.riesgo_por_segmento(datos.segmentos, dimension)
    if tabla.empty:
        return []
    if "Índice de riesgo" in tabla.columns:
        return asis.segmentos_empatados(tabla)
    return [str(tabla.iloc[0]["Segmento"])]


def _menor_segmento(datos, dimension) -> list[str]:
    tabla = rep.riesgo_por_segmento(datos.segmentos, dimension)
    return [] if tabla.empty else [str(tabla.iloc[-1]["Segmento"])]


def _top_cliente(datos) -> str | None:
    if datos.clientes.empty or "indice_riesgo_consolidado" not in datos.clientes.columns:
        return None
    return str(datos.clientes.sort_values("indice_riesgo_consolidado", ascending=False).iloc[0]["user_id"])


def _cliente_hechos(datos) -> list[str]:
    cliente = _top_cliente(datos)
    if cliente is None:
        return []
    fila = datos.clientes[datos.clientes["user_id"] == cliente].iloc[0]
    hechos = [f"{fila['indice_riesgo_consolidado']:.0f} de 100"]
    if pd.notna(fila.get("riesgo_score")):
        hechos.append(rep.fmt_pct(fila["riesgo_score"] * 100))
    return hechos


def _cliente_razones(datos) -> list[str]:
    """Para "¿por que es riesgoso?" tambien vale responder con sus factores de riesgo (S19)."""
    cliente = _top_cliente(datos)
    if cliente is None:
        return []
    razones = datos.clientes.loc[datos.clientes["user_id"] == cliente, "razones_riesgo"] \
        if "razones_riesgo" in datos.clientes.columns else pd.Series(dtype=str)
    primera = str(razones.iloc[0]).split(";")[0].strip() if not razones.empty and isinstance(razones.iloc[0], str) else ""
    return _cliente_hechos(datos) + ([primera] if primera else [])


def _escenario(datos, clave, campo) -> list[str]:
    sim = datos.simulacion_cartera
    if sim.empty or clave not in set(sim["escenario"]):
        return []
    return [rep.fmt_usd(sim.set_index("escenario").loc[clave, campo])]


def _demo(datos, clave) -> list[str]:
    sim = datos.simulacion_cliente
    if sim.empty or clave not in set(sim["escenario"]):
        return []
    return [rep.fmt_pct(sim.set_index("escenario").loc[clave, "prob_quedarse_sin_ahorro_pct"])]


def _diagnostico(datos) -> list[str]:
    resumen = rep.resumen_ejecutivo(datos.kpis, datos.segmentos)
    return [] if resumen is None else [resumen["diagnostico"]["nivel"], rep.fmt_int(resumen["total_clientes"])]


def _pronostico(datos, series=("ingreso", "gasto")) -> list[str]:
    if datos.pronostico.empty:
        return []
    return [rep.fmt_usd(datos.pronostico.query(f"serie == '{serie}' and tipo == 'pronostico'")["valor"].mean())
            for serie in series]


def banco_preguntas(datos: asis.DatosAsistente) -> list[Caso]:
    """50 preguntas (S19; antes 30): con 30, un solo fallo movia el resultado 3.3 puntos."""
    cliente = _top_cliente(datos) or "U00001"
    return [
        # --- Resumen ---
        Caso("¿Cómo está la cartera hoy?", "resumen", _diagnostico),
        Caso("Dame un resumen de la salud de la cartera", "resumen", _diagnostico),
        Caso("¿Cuál es la situación general de la cartera?", "resumen", _diagnostico),
        # --- Indicadores ---
        Caso("¿Cuál es la pérdida esperada?", "indicadores", lambda d: _fmt(d, "perdida_esperada_usd")),
        Caso("¿Cuál es la estimación de pérdida este año?", "indicadores", lambda d: _fmt(d, "perdida_esperada_usd")),
        Caso("¿Qué porcentaje de lo prestado esperamos perder?", "indicadores", lambda d: _fmt(d, "perdida_esperada_pct_exposicion")),
        Caso("¿Cuánto dinero tenemos prestado?", "indicadores", lambda d: _fmt(d, "exposicion_total_usd")),
        Caso("¿Cuál es la exposición total de la cartera?", "indicadores", lambda d: _fmt(d, "exposicion_total_usd")),
        Caso("¿Cuál es la capacidad de ahorro de los clientes?", "indicadores", lambda d: _fmt(d, "capacidad_ahorro_mediana_pct")),
        # Responder "no se cumple / requiere atencion" sin la cifra tambien es correcto
        Caso("¿Se cumple la meta de ahorro?", "indicadores", lambda d: _fmt(d, "capacidad_ahorro_mediana_pct") + _estado(d, "capacidad_ahorro_mediana_pct")),
        Caso("¿Cuál es el endeudamiento mensual?", "indicadores", lambda d: _fmt(d, "endeudamiento_mensual_mediana_pct")),
        Caso("¿Qué tan endeudados están los clientes?", "indicadores", lambda d: _fmt(d, "endeudamiento_mensual_mediana_pct")),
        Caso("¿Se cumple la meta de endeudamiento?", "indicadores", lambda d: _fmt(d, "endeudamiento_mensual_mediana_pct") + _estado(d, "endeudamiento_mensual_mediana_pct")),
        Caso("¿Qué porcentaje de clientes está en rojo?", "indicadores", lambda d: _fmt(d, "pct_clientes_semaforo_alto")),
        Caso("¿Cuál es el índice de riesgo promedio?", "indicadores", lambda d: _fmt(d, "indice_riesgo_consolidado_promedio")),
        Caso("¿Cuál es la probabilidad de impago promedio?", "indicadores", lambda d: _fmt(d, "probabilidad_incumplimiento_promedio_pct")),
        Caso("¿Qué porcentaje de clientes tiene alto riesgo de impago?", "indicadores", lambda d: _fmt(d, "probabilidad_incumplimiento_promedio_pct") + _fmt(d, "pct_clientes_riesgo_alto")),
        Caso("¿Cuál es el credit score promedio?", "indicadores", lambda d: _fmt(d, "credit_score_promedio")),
        Caso("¿Cuántos clientes son subprime?", "indicadores", lambda d: _fmt(d, "pct_clientes_subprime")),
        Caso("¿Qué tan preciso es el modelo de riesgo?", "indicadores", lambda d: _fmt(d, "auc_modelo_riesgo")),
        Caso("¿Cuántos clientes gastan más de lo que ganan (sin ahorro)?", "indicadores", lambda d: _fmt(d, "pct_clientes_deficitarios")),
        Caso("¿Qué tan concentrada está la cartera?", "indicadores", lambda d: _fmt(d, "indice_concentracion_region_hhi")),
        Caso("¿Cuántos clientes tienen déficit?", "indicadores", lambda d: _fmt(d, "pct_clientes_deficitarios")),
        # --- Segmentos ---
        Caso("¿Qué región tiene más riesgo?", "segmentos", lambda d: _top_segmento(d, "region")),
        Caso("¿En qué zona está el mayor riesgo?", "segmentos", lambda d: _top_segmento(d, "region")),
        Caso("¿Qué región tiene menos riesgo?", "segmentos", lambda d: _menor_segmento(d, "region")),
        Caso("¿Qué grupo de edad es más riesgoso?", "segmentos", lambda d: _top_segmento(d, "rango_edad")),
        Caso("¿Qué situación laboral tiene más riesgo?", "segmentos", lambda d: _top_segmento(d, "employment_status")),
        Caso("¿Qué tipo de préstamo es más riesgoso?", "segmentos", lambda d: _top_segmento(d, "tipo_prestamo")),
        Caso("¿Qué calificación crediticia tiene más riesgo?", "segmentos", lambda d: _top_segmento(d, "rango_credit_score")),
        # --- Clientes ---
        Caso("¿A qué clientes debemos llamar primero?", "prioritarios", lambda d: [_top_cliente(d)] if _top_cliente(d) else []),
        Caso("¿Quiénes son los clientes prioritarios?", "prioritarios", lambda d: [_top_cliente(d)] if _top_cliente(d) else []),
        Caso("¿A quién debemos contactar esta semana?", "prioritarios", lambda d: [_top_cliente(d)] if _top_cliente(d) else []),
        Caso(f"¿Por qué el cliente {cliente} es riesgoso?", "cliente", _cliente_razones),
        Caso(f"¿Cuál es la probabilidad de impago del cliente {cliente}?", "cliente", _cliente_hechos),
        Caso(f"Dame el detalle del cliente {cliente}", "cliente", _cliente_hechos),
        # --- Escenarios ---
        Caso("¿Qué pasaría si hay una crisis?", "escenarios", lambda d: _escenario(d, "severo", "perdida_esperada")),
        Caso("¿Cuánto perderíamos en un escenario severo?", "escenarios", lambda d: _escenario(d, "severo", "perdida_esperada")),
        Caso("¿Cuánto perderíamos en un escenario adverso?", "escenarios", lambda d: _escenario(d, "adverso", "perdida_esperada")),
        Caso("¿Cuál es la pérdida esperada en el escenario base?", "escenarios", lambda d: _escenario(d, "base", "perdida_esperada")),
        Caso("¿Nos alcanza el colchón en el escenario adverso?", "escenarios",
             lambda d: [] if d.simulacion_cartera.empty else
             [rep.fmt_pct(d.simulacion_cartera.set_index("escenario").loc["adverso", "prob_superar_colchon_pct"])]
             + _escenario(d, "adverso", "var_99_9")),
        Caso("¿Cuál es el peor caso de pérdida?", "escenarios",
             lambda d: _escenario(d, "base", "var_99_9") + _escenario(d, "severo", "var_99_9")),
        Caso("¿De cuánto es el colchón de capital?", "escenarios", lambda d: _escenario(d, "base", "var_99_9")),
        Caso("¿Qué dice la simulación Monte Carlo?", "escenarios",
             lambda d: _escenario(d, "base", "var_99_9") + _escenario(d, "severo", "perdida_esperada")),
        # --- Cliente demo ---
        Caso("¿El cliente demo aguanta perder su empleo?", "cliente_demo", lambda d: _demo(d, "desempleo")),
        Caso("¿Qué pasa con el cliente demo si hay inflación?", "cliente_demo", lambda d: _demo(d, "inflacion")),
        Caso("¿Qué le pasa al cliente demo en una recesión?", "cliente_demo", lambda d: _demo(d, "recesion")),
        Caso("¿Cuál es el pronóstico del cliente demo?", "cliente_demo", _pronostico),
        Caso("¿Cuánto ganará el cliente demo los próximos meses?", "cliente_demo", lambda d: _pronostico(d, ("ingreso",))),
        Caso("¿Cuánto gastará el cliente demo los próximos meses?", "cliente_demo", lambda d: _pronostico(d, ("gasto",))),
    ]


def contiene_hecho(respuesta: str, hechos: list[str]) -> bool:
    """Un hecho con cifras cuenta si TODAS sus cifras aparecen en la respuesta;
    uno de texto, si aparece (sin importar acentos ni mayusculas)."""
    numeros = asis._numeros(respuesta)
    texto = asis.normalizar(respuesta)
    for hecho in hechos:
        cifras = asis._numeros(hecho)
        if cifras and cifras <= numeros:
            return True
        if not cifras and asis.normalizar(hecho) in texto:
            return True
    return False


def se_nego(respuesta: str) -> bool:
    return any(f in asis.normalizar(respuesta) for f in asis.FRASES_SIN_INFORMACION)


def evaluar(datos: asis.DatosAsistente, llm, casos: list[Caso] | None = None,
            repeticiones: int = 1) -> tuple[pd.DataFrame, dict]:
    """`repeticiones` > 1 corre el banco varias veces y promedia (S19): mide la
    variacion entre corridas en lugar de depender de una sola."""
    filas = []
    for ronda in range(1, repeticiones + 1):
        for caso in casos or banco_preguntas(datos):
            hechos = caso.hechos(datos)
            inicio = time.time()
            resultado = asis.responder(caso.pregunta, datos, llm)
            rutas = [c for c, _ in asis.enrutar(caso.pregunta)]
            final = resultado["respuesta"]
            filas.append({
                "ronda": ronda,
                "pregunta": caso.pregunta,
                "enrutamiento_ok": caso.consulta in rutas,
                "modo": resultado["modo"],
                "ia_a_la_primera": resultado["modo"] == "ia" and not resultado["problemas"],
                "veraz_final": not asis.cifras_no_respaldadas(final, resultado["contexto"], caso.pregunta),
                "exito_ia": resultado["modo"] == "ia" and bool(hechos) and contiene_hecho(final, hechos) and not se_nego(final),
                "exito_final": bool(hechos) and contiene_hecho(final, hechos) and not se_nego(final),
                "segundos": round(time.time() - inicio, 1),
                "problemas": "; ".join(resultado["problemas"]),
                "respuesta": final[:300],
            })
    tabla = pd.DataFrame(filas)
    n = len(tabla)

    def pct(columna):
        return round(float(tabla[columna].mean()) * 100, 1) if n else 0.0

    resumen = {
        "preguntas": n,
        "enrutamiento_pct": pct("enrutamiento_ok"),
        "veracidad_ia_primer_intento_pct": pct("ia_a_la_primera"),
        "veracidad_final_pct": pct("veraz_final"),
        "exito_ia_pct": pct("exito_ia"),
        "exito_final_pct": pct("exito_final"),
        "respuestas_redactadas_por_ia_pct": round(float((tabla["modo"] == "ia").mean()) * 100, 1) if n else 0.0,
        "segundos_promedio": round(float(tabla["segundos"].mean()), 1) if n else 0.0,
        "meta_pct": META_PCT,
    }
    resumen["repeticiones"] = int(tabla["ronda"].max()) if n else 0
    resumen["exito_final_por_ronda"] = [round(float(v) * 100, 1) for v in tabla.groupby("ronda")["exito_final"].mean()]
    resumen["cumple_meta"] = bool(all(resumen[k] >= META_PCT for k in ["veracidad_final_pct", "exito_final_pct"]))
    return tabla, resumen


def cargar_datos(fecha: str | None) -> asis.DatosAsistente:
    """Mismas tablas que usa el dashboard ejecutivo (data_loader esta en /app)."""
    from data_loader import (cargar_clientes_particion, cargar_gold_kpis, cargar_gold_kpis_segmento,
                             cargar_tabla_particion)

    kpis = cargar_gold_kpis()
    fecha = fecha or sorted(kpis["fecha"].unique())[-1]
    segmentos = cargar_gold_kpis_segmento()
    return asis.DatosAsistente(
        fecha=fecha,
        kpis=kpis[kpis["fecha"] == fecha],
        segmentos=segmentos[segmentos["fecha"] == fecha] if not segmentos.empty else segmentos,
        clientes=cargar_clientes_particion(fecha),
        simulacion_cartera=cargar_tabla_particion("gold_simulacion_cartera", fecha),
        simulacion_cliente=cargar_tabla_particion("gold_simulacion_cliente_demo_resumen", fecha),
        pronostico=cargar_tabla_particion("gold_pronostico_cliente_demo", fecha),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el asistente financiero contra un banco de preguntas.")
    parser.add_argument("--fecha", help="Particion Gold AAAA/MM/DD (por defecto la mas reciente)")
    parser.add_argument("--modelo", default=asis.OLLAMA_MODEL)
    parser.add_argument("--repeticiones", type=int, default=1, help="Corridas del banco completo (se promedian)")
    parser.add_argument("--salida", default=os.path.join(CARPETA_REPORTES, "evaluacion_asistente.csv"))
    args = parser.parse_args()

    disponible, detalle = asis.ollama_disponible(modelo=args.modelo)
    if not disponible:
        raise SystemExit(f"Ollama no disponible: {detalle}")
    datos = cargar_datos(args.fecha)
    # Se carga el modelo antes de medir (el dashboard tambien lo calienta al abrirse);
    # la carga se reporta aparte.
    inicio = time.time()
    asis.calentar_modelo(modelo=args.modelo)
    carga = round(time.time() - inicio, 1)
    print(f"Modelo {args.modelo} cargado en memoria en {carga} s.")
    print(f"Evaluando con la fecha {datos.fecha}, {args.repeticiones} ronda(s)... (puede tardar varios minutos)")
    tabla, resumen = evaluar(datos, asis.crear_llm(args.modelo), repeticiones=args.repeticiones)
    resumen["segundos_carga_modelo"] = carga
    os.makedirs(os.path.dirname(os.path.abspath(args.salida)), exist_ok=True)
    tabla.to_csv(args.salida, index=False)
    # Semana 18: el resumen queda para la Vista 0 (SLA "Exactitud del asistente")
    with open(os.path.join(os.path.dirname(os.path.abspath(args.salida)), "evaluacion_asistente.json"), "w") as f:
        json.dump({**resumen, "modelo": args.modelo, "fecha_datos": datos.fecha,
                   "evaluado_en": pd.Timestamp.now(tz="America/Mexico_City").isoformat()}, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(tabla[["ronda", "pregunta", "enrutamiento_ok", "modo", "ia_a_la_primera", "exito_final", "segundos"]].to_string(index=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Detalle completo (respuestas y problemas): {args.salida}")


if __name__ == "__main__":
    main()
