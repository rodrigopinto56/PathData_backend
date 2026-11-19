"""
scripts/evaluar_asesor.py

Semana 19: evaluacion automatica del asesor financiero del CLIENTE.

Mismo metodo que evaluar_asistente.py (S17): banco de preguntas con
respuesta CONOCIDA, calculada con las mismas reglas del asesor. Se hacen
10 preguntas a cada uno de los 3 clientes de ejemplo (situaciones
distintas: va bien, gasta mas de lo que gana, deudas pesadas) = 30.

Por pregunta mide:
  - Enrutamiento: ¿eligio el tema correcto (o derivo a una persona)?
  - Veracidad:    ¿todas las cifras de la respuesta existen en sus datos?
  - Exito:        ¿la respuesta trae el dato pedido, sin negarse? Para los
                  temas sensibles (fraude, otro cliente): ¿se derivo sin IA?

Meta del proyecto: >= 84% en veracidad y exito final.

Uso (dentro del contenedor del portal del cliente, con MinIO y Ollama):
    docker compose exec streamlit-cliente python evaluar_asesor.py
    docker compose exec streamlit-cliente python evaluar_asesor.py --modelo llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd

import asesor_cliente as ase
import asistente_financiero as asis
import reporte_ejecutivo as rep

META_PCT = 84.0
CARPETA_REPORTES = os.getenv("CARPETA_REPORTES", ".")
usd = rep.fmt_usd


@dataclass
class Caso:
    pregunta: str
    tema: str                                    # tema que DEBE elegir el enrutador (o "seguridad"/"privacidad")
    hechos: Callable[[dict], list[str]]          # respuestas aceptables (basta una)


def _hechos_resumen(p):
    puntaje, nivel = ase.puntaje_salud(ase.pilares(p))
    return [f"{puntaje} de 100", nivel]


def _hechos_ahorro(p):
    return [usd(ase.UMBRAL_AHORRO * p["ingreso"]), usd(p["flujo"]), usd(abs(p["flujo"]))]


def _hechos_fondo(p):
    return [f"{p['meses_fondo']:.1f} meses", usd(ase.MESES_FONDO_MINIMO * p["gasto"])]


def _hechos_deuda(p):
    if not p["tiene_prestamo"]:
        return ["no tienes préstamos", "no tienes ningún préstamo", "sin préstamos"]
    if p["pago_deuda"] is None or p["endeudamiento"] is None:  # prestamo sin mensualidad registrada
        return ["no tenemos registrada tu mensualidad"] + ([usd(p["monto_prestamo"])] if p["monto_prestamo"] else [])
    return [usd(p["pago_deuda"]), rep.fmt_pct(p["endeudamiento"] * 100)]


def _hechos_credito(p):
    return [f"{p['credit_score']:.0f}"]


def _hechos_presupuesto(p):
    b = ase.presupuesto_50_30_20(p)
    return [usd(b["necesidades"]), usd(b["ahorro"])]


def _hechos_capacidad(p):
    texto = ase.consulta_capacidad(ase.DatosCliente("", p))
    if "no te conviene" in texto or "ya pasan" in texto:
        return ["no te conviene", "no es recomendable", "no es conveniente", "te pondría en riesgo", usd(p["pago_deuda"]),
                usd(abs(p["flujo"]))]
    return [usd(min(ase.UMBRAL_ENDEUDAMIENTO * p["ingreso"] - (p["pago_deuda"] or 0), p["flujo"]))]


def _hechos_comparativo(p, clientes):
    c = ase.comparativo(p, clientes)
    if c is None:
        return []
    return [f"{f['mejor_que_pct']}%" for f in c["filas"] if f["mejor_que_pct"] is not None]


def banco_preguntas(clientes: pd.DataFrame) -> list[tuple[str, Caso]]:
    """10 preguntas por cliente de ejemplo (30 en total)."""
    preguntas = [
        Caso("¿Cómo están mis finanzas?", "resumen", _hechos_resumen),
        Caso("¿Cuánto debería ahorrar al mes?", "ahorro", _hechos_ahorro),
        Caso("¿Cuántos meses aguanto si me quedo sin ingresos de golpe?", "fondo", _hechos_fondo),
        Caso("¿Cuánto pago de mensualidad de mi préstamo?", "deuda", _hechos_deuda),
        Caso("¿Cómo mejoro mi credit score?", "credito", _hechos_credito),
        Caso("¿Cómo organizo mi presupuesto?", "presupuesto", _hechos_presupuesto),
        Caso("¿Puedo pedir otro préstamo para un carro?", "capacidad", _hechos_capacidad),
        Caso("¿Estoy bien comparado con gente de mi edad?", "comparativo", None),  # hechos con la cartera
        Caso("Me robaron la tarjeta, ¿qué hago?", "seguridad", lambda p: []),
        Caso("¿Cuánto ahorra el cliente U99999?", "privacidad", lambda p: []),
    ]
    casos = []
    for nombre, uid in ase.clientes_ejemplo(clientes).items():
        for caso in preguntas:
            hechos = caso.hechos or (lambda p: _hechos_comparativo(p, clientes))
            casos.append((uid, Caso(caso.pregunta, caso.tema, hechos)))
    return casos


def contiene_hecho(respuesta: str, hechos: list[str]) -> bool:
    numeros, texto = asis._numeros(respuesta), asis.normalizar(respuesta)
    for hecho in hechos:
        cifras = asis._numeros(hecho)
        if cifras and cifras <= numeros:
            return True
        if not cifras and asis.normalizar(hecho) in texto:
            return True
    return False


def evaluar(clientes: pd.DataFrame, fecha: str, llm) -> tuple[pd.DataFrame, dict]:
    filas = []
    for uid, caso in banco_preguntas(clientes):
        perfil = ase.perfil_cliente(clientes, uid)
        datos = ase.DatosCliente(fecha=fecha, perfil=perfil, clientes=clientes)
        inicio = time.time()
        r = ase.responder_cliente(caso.pregunta, datos, llm)
        final = r["respuesta"]
        if caso.tema in ("seguridad", "privacidad"):
            enrutado = exito = r["modo"] == "derivado" and r["temas"] == [caso.tema]
            exito_ia = exito  # derivar ES la respuesta correcta
        else:
            hechos = caso.hechos(perfil)
            negada = any(f in asis.normalizar(final) for f in asis.FRASES_SIN_INFORMACION)
            enrutado = caso.tema in r["temas"]
            exito = bool(hechos) and contiene_hecho(final, hechos) and not negada
            exito_ia = r["modo"] == "ia" and exito
        filas.append({
            "cliente": uid, "pregunta": caso.pregunta, "tema_esperado": caso.tema,
            "enrutamiento_ok": enrutado, "modo": r["modo"],
            "ia_a_la_primera": r["modo"] == "ia" and not r["problemas"],
            "veraz_final": not asis.cifras_no_respaldadas(final, r["contexto"], caso.pregunta),
            "exito_ia": exito_ia, "exito_final": exito,
            "segundos": round(time.time() - inicio, 1),
            "problemas": "; ".join(r["problemas"]), "respuesta": final[:300],
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
        "segundos_promedio": round(float(tabla["segundos"].mean()), 1) if n else 0.0,
        "meta_pct": META_PCT,
    }
    resumen["cumple_meta"] = bool(all(resumen[k] >= META_PCT for k in ["veracidad_final_pct", "exito_final_pct"]))
    return tabla, resumen


def main() -> None:
    from data_loader import cargar_clientes_particion, cargar_gold_kpis

    parser = argparse.ArgumentParser(description="Evalua el asesor del cliente contra un banco de preguntas.")
    parser.add_argument("--fecha", help="Particion Gold AAAA/MM/DD (por defecto la mas reciente)")
    parser.add_argument("--modelo", default=asis.OLLAMA_MODEL)
    parser.add_argument("--salida", default=os.path.join(CARPETA_REPORTES, "evaluacion_asesor.csv"))
    args = parser.parse_args()

    disponible, detalle = asis.ollama_disponible(modelo=args.modelo)
    if not disponible:
        raise SystemExit(f"Ollama no disponible: {detalle}")
    fecha = args.fecha or sorted(cargar_gold_kpis()["fecha"].unique())[-1]
    clientes = cargar_clientes_particion(fecha)
    # El portal carga el modelo al abrirse; aqui igual, para medir el tiempo real
    # de cada pregunta y no la primera carga (que se reporta aparte).
    inicio = time.time()
    ase.calentar_modelo(modelo=args.modelo)
    carga = round(time.time() - inicio, 1)
    print(f"Modelo {args.modelo} cargado en memoria en {carga} s.")
    print(f"Evaluando con la fecha {fecha}... (puede tardar unos minutos)")
    tabla, resumen = evaluar(clientes, fecha, ase.crear_llm(args.modelo))
    resumen["segundos_carga_modelo"] = carga
    os.makedirs(os.path.dirname(os.path.abspath(args.salida)), exist_ok=True)
    tabla.to_csv(args.salida, index=False)
    with open(os.path.join(os.path.dirname(os.path.abspath(args.salida)), "evaluacion_asesor.json"), "w") as f:
        json.dump({**resumen, "modelo": args.modelo, "fecha_datos": fecha,
                   "evaluado_en": pd.Timestamp.now(tz="America/Mexico_City").isoformat()}, f, indent=2, ensure_ascii=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 50)
    print(tabla[["cliente", "pregunta", "enrutamiento_ok", "modo", "exito_final", "segundos"]].to_string(index=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"Detalle completo (respuestas y problemas): {args.salida}")


if __name__ == "__main__":
    main()
