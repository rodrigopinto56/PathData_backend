"""
scripts/prediccion_financiera.py

Semana 15: prediccion financiera del cliente demo + KPI de estabilidad.

Fuente: transacciones_cliente_demo (Silver), 60 meses de UN cliente
(kaggle ramyapintchy/personal-finance-data). No representa a la cartera:
es el "cliente demo" del gemelo digital, y asi se etiqueta en todo lado.

Que hace:
  1) Agrega las transacciones a series mensuales: ingreso, gasto, ahorro.
  2) Compara modelos de pronostico SIMPLES (con 60 puntos no se justifica
     nada mas complejo) contra baselines, con backtesting de origen movil:
     se "viaja al pasado" 12 veces, se pronostican los 6 meses siguientes
     y se compara contra lo que realmente paso.
  3) Elige el modelo de menor error. Si ninguno le gana al baseline
     ingenuo, se usa el baseline (y se dice).
  4) Pronostica 6 meses con banda de incertidumbre (percentiles 10-90 de
     los errores del backtest).
  5) Calcula la estabilidad de ingresos y gastos (coeficiente de variacion
     de 3 y 6 meses) -> KPI "Estabilidad de ingresos y gastos".

Sin dependencias nuevas (numpy/pandas). E/S solo en pronosticar_particion().
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HORIZONTE = 6
ORIGENES_BACKTEST = 12
UMBRAL_ESTABILIDAD_CV = 30.0  # % -- coeficiente de variacion mensual por debajo = estable
SERIES = ["ingreso", "gasto"]


# ---------------------------------------------------------------------------
# 1) Series mensuales
# ---------------------------------------------------------------------------
def serie_mensual(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Transacciones -> una fila por mes con ingreso, gasto y ahorro.
    Meses sin movimientos cuentan como 0; el ultimo mes se descarta si
    esta incompleto (sus datos terminan antes del dia 20)."""
    datos = df_tx.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos["mes"] = datos["fecha"].dt.to_period("M")
    tabla = datos.pivot_table(index="mes", columns="tipo", values="monto", aggfunc="sum").fillna(0.0)
    meses = pd.period_range(tabla.index.min(), tabla.index.max(), freq="M")
    tabla = tabla.reindex(meses, fill_value=0.0)
    serie = pd.DataFrame(
        {"ingreso": tabla.get("Income", 0.0), "gasto": tabla.get("Expense", 0.0)}, index=meses
    )
    if datos["fecha"].max().day < 20:
        serie = serie.iloc[:-1]
    serie["ahorro"] = serie["ingreso"] - serie["gasto"]
    serie.index.name = "mes"
    return serie.round(2)


def resumen_por_categoria(df_tx: pd.DataFrame) -> pd.DataFrame:
    """gold_transacciones_mensuales: mes x tipo x categoria."""
    datos = df_tx.copy()
    datos["mes"] = pd.to_datetime(datos["fecha"]).dt.to_period("M").astype(str)
    return (
        datos.groupby(["mes", "tipo", "categoria"])["monto"]
        .agg(monto="sum", transacciones="count")
        .reset_index()
        .round(2)
    )


# ---------------------------------------------------------------------------
# 2) Modelos de pronostico (todos: historia -> h valores)
# ---------------------------------------------------------------------------
def naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h)


def naive_estacional(y: np.ndarray, h: int) -> np.ndarray:
    if len(y) < 12:
        return naive(y, h)
    return np.array([y[-12 + (i % 12)] for i in range(h)])


def media_movil(ventana: int):
    def modelo(y: np.ndarray, h: int) -> np.ndarray:
        return np.repeat(np.mean(y[-ventana:]), h)
    modelo.__name__ = f"media_movil_{ventana}"
    return modelo


def suavizamiento_simple(y: np.ndarray, h: int) -> np.ndarray:
    """Suavizamiento exponencial simple; alpha elegido por error a 1 paso."""
    mejor, nivel_final = np.inf, y[0]
    for alpha in np.linspace(0.05, 0.95, 19):
        nivel, error = y[0], 0.0
        for valor in y[1:]:
            error += (valor - nivel) ** 2
            nivel = alpha * valor + (1 - alpha) * nivel
        if error < mejor:
            mejor, nivel_final = error, nivel
    return np.repeat(nivel_final, h)


def holt_amortiguado(y: np.ndarray, h: int, phi: float = 0.9) -> np.ndarray:
    """Holt con tendencia amortiguada (la tendencia se apaga con el tiempo:
    evita proyectar crecimientos infinitos con pocos datos)."""
    mejor, estado = np.inf, (y[0], 0.0)
    for alpha in np.linspace(0.1, 0.9, 9):
        for beta in np.linspace(0.05, 0.5, 10):
            nivel, tendencia, error = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0, 0.0
            for valor in y[1:]:
                prediccion = nivel + phi * tendencia
                error += (valor - prediccion) ** 2
                nivel_nuevo = alpha * valor + (1 - alpha) * prediccion
                tendencia = beta * (nivel_nuevo - nivel) + (1 - beta) * phi * tendencia
                nivel = nivel_nuevo
            if error < mejor:
                mejor, estado = error, (nivel, tendencia)
    nivel, tendencia = estado
    return np.array([nivel + sum(phi ** k for k in range(1, i + 2)) * tendencia for i in range(h)])


MODELOS = {
    "naive": naive,
    "naive_estacional": naive_estacional,
    "media_movil_3": media_movil(3),
    "media_movil_6": media_movil(6),
    "suavizamiento_simple": suavizamiento_simple,
    "holt_amortiguado": holt_amortiguado,
}
BASELINE = "naive"


# ---------------------------------------------------------------------------
# 3) Backtesting y seleccion
# ---------------------------------------------------------------------------
def backtest(y: np.ndarray, modelo, h: int = HORIZONTE, origenes: int = ORIGENES_BACKTEST) -> np.ndarray:
    """Matriz de errores (real - pronostico): una fila por origen, una columna por horizonte."""
    n = len(y)
    primero = max(12, n - origenes - h + 1)
    errores = [y[t:t + h] - modelo(y[:t], h) for t in range(primero, n - h + 1)]
    return np.array(errores)


def comparar_modelos(y: np.ndarray) -> pd.DataFrame:
    filas = []
    for nombre, modelo in MODELOS.items():
        errores = backtest(y, modelo)
        filas.append({"modelo": nombre, "mae": float(np.mean(np.abs(errores))), "origenes": len(errores)})
    tabla = pd.DataFrame(filas).sort_values("mae").reset_index(drop=True)
    mae_baseline = float(tabla.loc[tabla["modelo"] == BASELINE, "mae"].iloc[0])
    tabla["mejora_vs_baseline_pct"] = ((1 - tabla["mae"] / mae_baseline) * 100).round(1)
    tabla["mae"] = tabla["mae"].round(2)
    return tabla


# ---------------------------------------------------------------------------
# 4) Pronostico con banda y 5) estabilidad
# ---------------------------------------------------------------------------
def estabilidad(serie: pd.DataFrame) -> dict:
    """Coeficiente de variacion (desviacion / promedio) de los ultimos 3 y 6
    meses: cuanto se mueve el ingreso y el gasto mes a mes."""
    resultado = {}
    for columna in SERIES:
        for meses in (3, 6):
            ventana = serie[columna].iloc[-meses:]
            media = ventana.mean()
            resultado[f"cv_{columna}_{meses}m_pct"] = round(float(ventana.std(ddof=1) / media * 100), 1) if media else None
        ultimos = serie[columna].iloc[-6:]
        pendiente = np.polyfit(np.arange(len(ultimos)), ultimos.to_numpy(), 1)[0] if len(ultimos) > 1 else 0.0
        resultado[f"tendencia_{columna}_pct_mensual"] = round(float(pendiente / ultimos.mean() * 100), 1) if ultimos.mean() else None
    peor = max(v for k, v in resultado.items() if k.endswith("_6m_pct") and v is not None)
    resultado["cv_6m_peor_pct"] = peor
    resultado["estable"] = bool(peor < UMBRAL_ESTABILIDAD_CV)
    return resultado


def pronosticar(serie: pd.DataFrame) -> dict:
    comparaciones, elegidos, pronosticos, errores = {}, {}, {}, {}
    for columna in SERIES:
        y = serie[columna].to_numpy(dtype=float)
        tabla = comparar_modelos(y)
        comparaciones[columna] = tabla
        mejor = tabla.iloc[0]
        # Solo se usa un modelo si le gana al baseline; si no, el baseline.
        elegido = mejor["modelo"] if mejor["mae"] < tabla.loc[tabla["modelo"] == BASELINE, "mae"].iloc[0] else BASELINE
        elegidos[columna] = elegido
        pronosticos[columna] = MODELOS[elegido](y, HORIZONTE)
        errores[columna] = backtest(y, MODELOS[elegido])
    # El ahorro es la diferencia (consistente con sus componentes); su banda
    # sale de los errores combinados del backtest.
    pronosticos["ahorro"] = pronosticos["ingreso"] - pronosticos["gasto"]
    errores["ahorro"] = errores["ingreso"] - errores["gasto"]

    meses_futuros = pd.period_range(serie.index[-1] + 1, periods=HORIZONTE, freq="M")
    filas = []
    for columna in ["ingreso", "gasto", "ahorro"]:
        for mes, valor in serie[columna].items():
            filas.append({"mes": str(mes), "serie": columna, "tipo": "historico", "valor": float(valor),
                          "limite_inferior": None, "limite_superior": None, "modelo": None})
        inferior = np.percentile(errores[columna], 10, axis=0)
        superior = np.percentile(errores[columna], 90, axis=0)
        for i, mes in enumerate(meses_futuros):
            valor = float(pronosticos[columna][i])
            filas.append({"mes": str(mes), "serie": columna, "tipo": "pronostico", "valor": round(valor, 2),
                          "limite_inferior": round(valor + float(inferior[i]), 2),
                          "limite_superior": round(valor + float(superior[i]), 2),
                          "modelo": elegidos.get(columna, f"{elegidos['ingreso']} - {elegidos['gasto']}")})
    return {
        "pronostico": pd.DataFrame(filas),
        "modelos_elegidos": elegidos,
        "comparacion_modelos": {c: t.to_dict(orient="records") for c, t in comparaciones.items()},
        "estabilidad": estabilidad(serie),
        "ultimo_mes_con_datos": str(serie.index[-1]),
        "meses_historia": int(len(serie)),
    }


# ---------------------------------------------------------------------------
# E/S con MinIO
# ---------------------------------------------------------------------------
def pronosticar_particion(fecha: str) -> dict:
    """Lee transacciones_cliente_demo (Silver) de `fecha`, pronostica y
    publica en Gold. Si la fuente opcional no esta, regresa disponible=False
    (el KPI de estabilidad queda como SIN_DATOS) en vez de fallar."""
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET, SILVER_BUCKET, STORAGE_OPTIONS

    gold = os.getenv("GOLD_BUCKET", "gold-layer")
    try:
        df_tx = pd.read_parquet(
            f"s3://{SILVER_BUCKET}/transacciones_cliente_demo/{fecha}/valid", storage_options=STORAGE_OPTIONS
        )
    except FileNotFoundError:
        return {"fecha": fecha, "disponible": False}

    serie = serie_mensual(df_tx)
    resultado = pronosticar(serie)
    resultado["pronostico"].to_parquet(
        f"s3://{gold}/gold_pronostico_cliente_demo/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS, index=False
    )
    resumen_por_categoria(df_tx).to_parquet(
        f"s3://{gold}/gold_transacciones_mensuales/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS, index=False
    )
    metadata = {
        "fecha": fecha,
        "disponible": True,
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "alcance": "cliente demo (1 cliente, 60 meses); no representa a la cartera",
        **{k: v for k, v in resultado.items() if k != "pronostico"},
    }
    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    with fs.open(f"{gold}/_pronostico_metadata/{fecha.replace('/', '-')}.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return {
        "fecha": fecha,
        "disponible": True,
        "modelos_elegidos": resultado["modelos_elegidos"],
        "cv_6m_peor_pct": resultado["estabilidad"]["cv_6m_peor_pct"],
    }


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(pronosticar_particion(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
