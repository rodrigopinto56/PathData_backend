"""
scripts/simulador_montecarlo.py

Semana 16: simulador financiero Monte Carlo.

Monte Carlo = en lugar de calcular UN resultado, se simulan miles de
futuros posibles con reglas probabilisticas y se mira la distribucion de
resultados (el promedio, pero sobre todo los casos malos).

Dos simuladores (configuracion: config/simulador.yml):

1) Perdidas de la cartera de credito -- modelo de un factor de Vasicek, el
   que usa Basilea II/III para el capital regulatorio de credito:
     - Cada simulacion = un ano. Se sortea un factor economico comun Z.
     - El prestamo i cae en impago si  sqrt(rho_i)*Z + sqrt(1-rho_i)*e_i < Phi^-1(PD_i)
       (e_i = suerte propia del cliente). Con Z malo, caen muchos a la vez.
     - Perdida del ano = suma de EAD x LGD de los que cayeron.
   Resultado: perdida esperada, VaR 95/99/99.9%, expected shortfall y
   capital economico, por escenario (base, adverso, severo).

2) Finanzas del cliente demo -- bootstrap de su historial mensual: cada
   simulacion arma 12 meses sorteando meses reales (ingreso y gasto del
   mismo mes juntos), aplica el escenario (inflacion, recesion, desempleo)
   y sigue su saldo ahorrado. Resultado: abanico de saldos y probabilidad
   de quedarse sin ahorro.

Funciones puras (numpy/pandas/scipy). La E/S esta en simular_particion().
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml
from scipy.special import ndtri  # inversa de la normal estandar (Phi^-1)

RUTAS_CONFIG = [
    "/opt/airflow/config/simulador.yml",
    "/app/simulador.yml",  # imagen del dashboard
    os.path.join(os.path.dirname(__file__), "..", "config", "simulador.yml"),
]
PERCENTILES_ABANICO = [5, 25, 50, 75, 95]


def cargar_config(ruta: str | None = None) -> dict:
    for candidata in ([ruta] if ruta else RUTAS_CONFIG):
        if candidata and os.path.exists(candidata):
            with open(candidata) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No se encontro config/simulador.yml en {RUTAS_CONFIG}")


# ---------------------------------------------------------------------------
# 1) Perdidas de la cartera (Vasicek)
# ---------------------------------------------------------------------------
def correlacion_basilea(pd_: np.ndarray) -> np.ndarray:
    """Correlacion de activos de Basilea II para credito minorista ("otros"):
    16% para PD muy bajas, bajando hasta 3% para PD altas."""
    peso = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
    return 0.03 * peso + 0.16 * (1 - peso)


def simular_perdidas(pd_: np.ndarray, ead: np.ndarray, lgd: float, correlacion,
                     simulaciones: int, semilla: int, lote: int = 250) -> np.ndarray:
    """Perdida total de la cartera en cada ano simulado (vector de largo
    `simulaciones`). Vectorizado por lotes para no crear una matriz enorme."""
    pd_ = np.clip(np.asarray(pd_, dtype=float), 1e-6, 1 - 1e-6)
    exposicion = np.asarray(ead, dtype=float) * lgd
    rho = correlacion_basilea(pd_) if correlacion in (None, "basilea") else np.full_like(pd_, float(correlacion))
    umbral = ndtri(pd_)
    peso_comun, peso_propio = np.sqrt(rho), np.sqrt(1 - rho)

    rng = np.random.default_rng(semilla)
    factor_economia = rng.standard_normal(simulaciones)
    perdidas = np.empty(simulaciones)
    for inicio in range(0, simulaciones, lote):
        z = factor_economia[inicio:inicio + lote, None]
        suerte_propia = rng.standard_normal((len(z), len(pd_)))
        impago = peso_comun * z + peso_propio * suerte_propia < umbral
        perdidas[inicio:inicio + lote] = impago @ exposicion
    return perdidas


def resumir_perdidas(perdidas: np.ndarray, exposicion_total: float, niveles: list[float]) -> dict:
    esperada = float(perdidas.mean())
    resumen = {
        "perdida_esperada": esperada,
        "perdida_esperada_pct": esperada / exposicion_total * 100 if exposicion_total else None,
        "desviacion": float(perdidas.std()),
    }
    for nivel in niveles:
        etiqueta = f"{nivel * 100:g}".replace(".", "_")
        var = float(np.quantile(perdidas, nivel))
        resumen[f"var_{etiqueta}"] = var
        resumen[f"var_{etiqueta}_pct"] = var / exposicion_total * 100 if exposicion_total else None
    var_99 = np.quantile(perdidas, 0.99)
    resumen["expected_shortfall_99"] = float(perdidas[perdidas >= var_99].mean())
    nivel_max = f"{max(niveles) * 100:g}".replace(".", "_")
    # Capital economico: lo que hay que tener ADEMAS de las reservas
    # (perdida esperada) para aguantar un ano de 1 en 1,000.
    resumen["capital_economico"] = resumen[f"var_{nivel_max}"] - esperada
    return resumen


def histograma(perdidas: np.ndarray, bins: int = 40) -> pd.DataFrame:
    frecuencia, bordes = np.histogram(perdidas, bins=bins)
    return pd.DataFrame({
        "desde": bordes[:-1], "hasta": bordes[1:], "anos": frecuencia,
        "pct_anos": frecuencia / frecuencia.sum() * 100,
    })


def preparar_cartera(fact: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """PD y exposicion (EAD) de los clientes con prestamo y PD calculada."""
    con_prestamo = (fact["tiene_prestamo"] == 1) & fact["riesgo_score"].notna() & (fact["loan_amount_usd"] > 0)
    datos = fact.loc[con_prestamo]
    return datos["riesgo_score"].to_numpy(dtype=float), datos["loan_amount_usd"].to_numpy(dtype=float)


def simular_escenario_cartera(pd_: np.ndarray, ead: np.ndarray, escenario: dict, config: dict,
                              simulaciones: int | None = None) -> tuple[dict, np.ndarray]:
    """Un escenario de estres: PD x multiplicador, LGD y correlacion propias."""
    pd_escenario = np.clip(pd_ * escenario.get("multiplicador_pd", 1.0), 0, 0.999)
    perdidas = simular_perdidas(
        pd_escenario, ead, escenario["lgd"], escenario.get("correlacion", config["correlacion"]),
        simulaciones or config["simulaciones"], config["semilla"],
    )
    exposicion_total = float(ead.sum())
    resumen = resumir_perdidas(perdidas, exposicion_total, config["niveles_var"])
    resumen.update({
        "exposicion_total": exposicion_total,
        "prestamos": int(len(ead)),
        "pd_promedio_pct": float(pd_escenario.mean() * 100),
        "lgd": escenario["lgd"],
        "simulaciones": int(len(perdidas)),
    })
    return resumen, perdidas


def simular_cartera(fact: pd.DataFrame, config: dict) -> dict:
    """Todos los escenarios configurados. El PRIMER escenario (base) fija el
    "colchon" del banco en condiciones normales:
      - reservas = su perdida esperada (lo que se aparta para perdidas normales)
      - reservas + capital = su VaR al nivel mas alto (aguanta 1 ano de 1,000)
    y en cada escenario se mide que tan seguido las perdidas los superan."""
    pd_, ead = preparar_cartera(fact)
    if len(ead) == 0:
        return {"disponible": False}
    nivel_max = f"var_{max(config['niveles_var']) * 100:g}".replace(".", "_")
    filas, histogramas, reservas, colchon = [], [], None, None
    for clave, escenario in config["escenarios"].items():
        resumen, perdidas = simular_escenario_cartera(pd_, ead, escenario, config)
        if reservas is None:
            reservas, colchon = resumen["perdida_esperada"], resumen[nivel_max]
        resumen["prob_superar_reservas_pct"] = float((perdidas > reservas).mean() * 100)
        resumen["prob_superar_colchon_pct"] = float((perdidas > colchon).mean() * 100)
        filas.append({"escenario": clave, "nombre": escenario["nombre"], "descripcion": escenario["descripcion"], **resumen})
        hist = histograma(perdidas)
        hist.insert(0, "escenario", clave)
        histogramas.append(hist)
    return {"disponible": True, "resumen": pd.DataFrame(filas), "distribucion": pd.concat(histogramas, ignore_index=True)}


# ---------------------------------------------------------------------------
# 2) Cliente demo (bootstrap de meses reales)
# ---------------------------------------------------------------------------
def simular_cliente(serie: pd.DataFrame, escenario: dict, config: dict,
                    saldo_inicial: float | None = None, simulaciones: int | None = None) -> dict:
    """`serie`: DataFrame mensual con columnas ingreso y gasto (historico).
    Regresa el abanico de saldo por mes y las probabilidades clave."""
    historia = serie[["ingreso", "gasto"]].iloc[-config["meses_historia"]:].to_numpy(dtype=float)
    n = simulaciones or config["simulaciones"]
    horizonte = config["horizonte_meses"]
    rng = np.random.default_rng(config["semilla"])

    meses = rng.integers(0, len(historia), size=(n, horizonte))  # meses completos: conserva ingreso-gasto
    ingreso = historia[meses, 0] * (1 + escenario.get("cambio_ingreso", 0.0))
    gasto = historia[meses, 1] * (1 + escenario.get("cambio_gasto", 0.0))
    sin_ingreso = int(escenario.get("meses_sin_ingreso", 0))
    if sin_ingreso:
        ingreso[:, :sin_ingreso] = 0.0

    if saldo_inicial is None:
        saldo_inicial = config["saldo_inicial_meses_gasto"] * historia[:, 1].mean()
    saldo = saldo_inicial + np.cumsum(ingreso - gasto, axis=1)

    negativo = saldo < 0
    se_agota = negativo.any(axis=1)
    mes_agotamiento = np.where(se_agota, negativo.argmax(axis=1) + 1, np.nan)
    abanico = pd.DataFrame({"mes": np.arange(1, horizonte + 1)})
    for p in PERCENTILES_ABANICO:
        abanico[f"p{p}"] = np.percentile(saldo, p, axis=0)
    abanico["prob_saldo_negativo_pct"] = negativo.mean(axis=0) * 100
    return {
        "abanico": abanico,
        "saldo_inicial": float(saldo_inicial),
        "prob_quedarse_sin_ahorro_pct": float(se_agota.mean() * 100),
        "mes_mediano_agotamiento": float(np.nanmedian(mes_agotamiento)) if se_agota.any() else None,
        "saldo_final_mediana": float(np.median(saldo[:, -1])),
        "flujo_mensual_promedio": float((ingreso - gasto).mean()),
        "simulaciones": int(n),
    }


def simular_cliente_escenarios(serie: pd.DataFrame, config: dict) -> dict:
    abanicos, resumen = [], []
    for clave, escenario in config["escenarios"].items():
        resultado = simular_cliente(serie, escenario, config)
        abanico = resultado.pop("abanico")
        abanico.insert(0, "escenario", clave)
        abanicos.append(abanico)
        resumen.append({"escenario": clave, "nombre": escenario["nombre"], "descripcion": escenario["descripcion"], **resultado})
    return {"abanico": pd.concat(abanicos, ignore_index=True), "resumen": pd.DataFrame(resumen)}


def serie_desde_pronostico(pronostico: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye la serie mensual historica desde gold_pronostico_cliente_demo
    (asi el dashboard no necesita leer Silver)."""
    historia = pronostico[(pronostico["tipo"] == "historico") & pronostico["serie"].isin(["ingreso", "gasto"])]
    return historia.pivot_table(index="mes", columns="serie", values="valor").sort_index()


# ---------------------------------------------------------------------------
# E/S con MinIO (la llama transformacion_gold_dag)
# ---------------------------------------------------------------------------
def simular_particion(fecha: str) -> dict:
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET, STORAGE_OPTIONS

    gold = os.getenv("GOLD_BUCKET", "gold-layer")
    config = cargar_config()
    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    inicio = time.time()
    metadata = {"fecha": fecha, "generado_en": datetime.now(timezone.utc).isoformat(), "config": config}

    def escribir(df: pd.DataFrame, tabla: str) -> None:
        df.to_parquet(f"s3://{gold}/{tabla}/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS, index=False)

    # --- 1) Cartera ---
    fact = pd.read_parquet(f"s3://{gold}/fact_posicion_financiera/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS)
    cartera = simular_cartera(fact, config["cartera"])
    metadata["cartera_disponible"] = cartera["disponible"]
    if cartera["disponible"]:
        escribir(cartera["resumen"], "gold_simulacion_cartera")
        escribir(cartera["distribucion"], "gold_simulacion_cartera_distribucion")
        metadata["cartera"] = cartera["resumen"].set_index("escenario")[
            ["perdida_esperada", "var_99", "capital_economico", "prob_superar_colchon_pct"]
        ].round(2).to_dict(orient="index")

    # --- 2) Cliente demo: la serie mensual que ya publico pronosticar_cliente_demo
    # (misma que ve el dashboard). Si la fuente opcional no esta, se omite.
    try:
        pronostico = pd.read_parquet(
            f"s3://{gold}/gold_pronostico_cliente_demo/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS
        )
    except FileNotFoundError:
        pronostico = None
    metadata["cliente_demo_disponible"] = pronostico is not None
    if pronostico is not None:
        cliente = simular_cliente_escenarios(serie_desde_pronostico(pronostico), config["cliente_demo"])
        escribir(cliente["abanico"], "gold_simulacion_cliente_demo")
        escribir(cliente["resumen"], "gold_simulacion_cliente_demo_resumen")
        metadata["cliente_demo"] = cliente["resumen"].set_index("escenario")[
            ["saldo_inicial", "prob_quedarse_sin_ahorro_pct", "saldo_final_mediana"]
        ].round(2).to_dict(orient="index")

    metadata["segundos"] = round(time.time() - inicio, 1)
    with fs.open(f"{gold}/_simulacion_metadata/{fecha.replace('/', '-')}.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return {k: metadata[k] for k in ["fecha", "cartera_disponible", "cliente_demo_disponible", "segundos"]}


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(simular_particion(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
