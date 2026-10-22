"""
scripts/features.py

Semana 14: preparacion de datasets para IA.

Contrato: config/features.yml. Este modulo:
  1) Calcula las MISMAS features para entrenamiento (credit_risk, con
     target) y para calificacion (personal_finance_ml, la cartera), usando
     ratios y posiciones relativas para que dos poblaciones distintas sean
     comparables.
  2) Mide el poder predictivo de cada candidata con WoE / Information
     Value (el estandar de credit scoring en banca).
  3) Selecciona variables con reglas explicitas (IV, redundancia, paridad,
     atributos protegidos) y deja un reporte de cada decision.
  4) Divide en entrenamiento / validacion / prueba (estratificado, semilla fija).
  5) Valida rangos y nulos contra el contrato.

Pandas puro, salvo construir_dataset_entrenamiento() (E/S a MinIO).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

RUTAS_CONTRATO = [
    "/opt/airflow/config/features.yml",
    os.path.join(os.path.dirname(__file__), "..", "config", "features.yml"),
]

GRADOS = ["A", "B", "C", "D", "E", "F", "G"]
TARGET = "impago"
# Distribucion de grados por defecto (aprox. la de credit_risk); se usa solo
# si todavia no hay un modelo entrenado que traiga la distribucion real.
DISTRIBUCION_GRADOS_DEFECTO = {"A": 0.33, "B": 0.32, "C": 0.20, "D": 0.11, "E": 0.03, "F": 0.007, "G": 0.003}


def cargar_contrato(ruta: str | None = None) -> dict:
    for candidata in ([ruta] if ruta else RUTAS_CONTRATO):
        if candidata and os.path.exists(candidata):
            with open(candidata) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No se encontro config/features.yml en {RUTAS_CONTRATO}")


def features_modelo(contrato: dict) -> list[str]:
    return [n for n, f in contrato["features"].items() if f.get("uso") == "modelo"]


# ---------------------------------------------------------------------------
# 1) Calculo de features (paridad entrenamiento / cartera)
# ---------------------------------------------------------------------------
def features_entrenamiento(df: pd.DataFrame, contrato: dict) -> pd.DataFrame:
    """credit_risk (Silver) -> features candidatas + target."""
    mapeo = contrato["features"]["proposito"]["mapeo_entrenamiento"]
    ingreso = df["person_income"].astype(float)
    monto = df["loan_amnt"].astype(float)
    return pd.DataFrame(
        {
            # --- candidatas del modelo ---
            "ratio_prestamo_ingreso": monto / ingreso.where(ingreso > 0),
            "ingreso_anual_log": np.log1p(ingreso),
            "monto_prestamo_log": np.log1p(monto),
            "tasa_interes": df["loan_int_rate"].astype(float),
            "grado_credito": df["loan_grade"].map({g: i + 1 for i, g in enumerate(GRADOS)}),
            "proposito": df["loan_intent"].map(mapeo).fillna("OTRO"),
            # --- candidatas evaluadas pero excluidas (ver contrato) ---
            "edad": df["person_age"].astype(float),
            "antiguedad_laboral": df["person_emp_length"].astype(float),
            "anios_historial_credito": df["cb_person_cred_hist_length"].astype(float),
            "impago_previo": (df["cb_person_default_on_file"] == "Y").astype(int),
            "renta_vivienda": (df["person_home_ownership"] == "RENT").astype(int),
            # --- target ---
            TARGET: df["loan_status"].astype(int),
        }
    ).reset_index(drop=True)


def grado_por_cuantiles(credit_score: pd.Series, distribucion: dict) -> pd.Series:
    """Asigna a cada cliente el grado que ocupa la misma posicion relativa
    que su credit_score: si en el entrenamiento el 33% son 'A', el 33% de
    scores mas altos de la cartera recibe 'A', y asi sucesivamente."""
    posicion = 1 - credit_score.rank(pct=True, method="average")  # 0 = mejor score
    acumulado = np.cumsum([distribucion[g] for g in GRADOS])
    acumulado = acumulado / acumulado[-1]
    indice = np.searchsorted(acumulado[:-1], posicion.to_numpy(), side="right")
    return pd.Series(indice + 1, index=credit_score.index).where(credit_score.notna())


def features_cartera(df: pd.DataFrame, contrato: dict, distribucion_grados: dict | None = None) -> pd.DataFrame:
    """personal_finance_ml (Silver) -> mismas features del modelo, mas
    columnas de identificacion y de auditoria (no entran al modelo)."""
    mapeo = contrato["features"]["proposito"]["mapeo_cartera"]
    ingreso_anual = df["monthly_income_usd"].astype(float) * 12
    tiene = df["has_loan"].astype(str).str.strip().str.lower().isin({"yes", "si", "sí", "true", "1"})
    monto = df["loan_amount_usd"].astype(float).where(tiene)
    tasa = df["loan_interest_rate_pct"].astype(float) if "loan_interest_rate_pct" in df.columns else np.nan
    proposito = df["loan_type"].map(mapeo).fillna("OTRO") if "loan_type" in df.columns else "OTRO"

    return pd.DataFrame(
        {
            "user_id": df["user_id"],
            "tiene_prestamo": tiene.astype(int),
            "ratio_prestamo_ingreso": monto / ingreso_anual.where(ingreso_anual > 0),
            "ingreso_anual_log": np.log1p(ingreso_anual),
            "monto_prestamo_log": np.log1p(monto),
            "tasa_interes": pd.Series(tasa, index=df.index).where(tiene),
            "grado_credito": grado_por_cuantiles(df["credit_score"].astype(float), distribucion_grados or DISTRIBUCION_GRADOS_DEFECTO),
            "proposito": pd.Series(proposito, index=df.index).where(tiene, "SIN_PRESTAMO"),
            # Solo para auditoria de sesgo (excluidas del modelo por politica)
            "genero": df["gender"] if "gender" in df.columns else None,
            "edad": df["age"] if "age" in df.columns else None,
        }
    ).reset_index(drop=True)


CUANTILES_REFERENCIA = np.linspace(0, 1, 101)


def cuantiles_referencia(df: pd.DataFrame, columnas: list[str]) -> dict:
    """Percentiles 0..100 de cada feature numerica en el entrenamiento."""
    return {c: [float(v) for v in np.nanquantile(df[c].astype(float), CUANTILES_REFERENCIA)] for c in columnas}


def alinear_por_cuantiles(df: pd.DataFrame, referencia: dict, mascara: pd.Series | None = None) -> pd.DataFrame:
    """Alineacion por cuantiles (quantile mapping): cada valor de la cartera
    se reemplaza por el valor del entrenamiento que ocupa el MISMO percentil.

    Por que: entrenamiento y cartera son poblaciones distintas (en la cartera
    el prestamo llega a varias veces el ingreso anual; en el entrenamiento
    nunca pasa de 0.83). Sin alinear, toda la cartera caeria en el extremo
    riesgoso del modelo. Alineado, el modelo compara a cada cliente contra
    su propia cartera (posicion relativa). El desalineamiento original se
    mide con PSI y queda en la ficha del modelo.

    `mascara` limita el calculo de percentiles a las filas relevantes (p.ej.
    clientes con prestamo); el resto queda igual."""
    salida = df.copy()
    filas = mascara if mascara is not None else pd.Series(True, index=df.index)
    for columna, valores_ref in referencia.items():
        if columna not in salida.columns:
            continue
        serie = salida.loc[filas, columna].astype(float)
        if serie.notna().sum() == 0:
            continue
        percentil = serie.rank(pct=True, method="average")
        salida.loc[filas, columna] = np.interp(percentil, CUANTILES_REFERENCIA, valores_ref).round(6)
        salida.loc[filas, columna] = salida.loc[filas, columna].where(serie.notna())
    return salida


def psi(referencia: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index: cuanto cambio una distribucion.
    < 0.1 estable, 0.1-0.25 cambio moderado, > 0.25 cambio fuerte."""
    referencia, actual = referencia.dropna().astype(float), actual.dropna().astype(float)
    if referencia.empty or actual.empty:
        return float("nan")
    cortes = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1)[1:-1]))
    p_ref = np.bincount(np.searchsorted(cortes, referencia), minlength=len(cortes) + 1) / len(referencia)
    p_act = np.bincount(np.searchsorted(cortes, actual), minlength=len(cortes) + 1) / len(actual)
    p_ref, p_act = np.clip(p_ref, 1e-4, None), np.clip(p_act, 1e-4, None)
    return float(((p_act - p_ref) * np.log(p_act / p_ref)).sum())


# ---------------------------------------------------------------------------
# 2) WoE / IV con binning monotono
# ---------------------------------------------------------------------------
def _es_numerica(serie: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(serie)


def _cortes_iniciales(x: pd.Series, n_bins: int) -> list[float]:
    valores = np.sort(x.dropna().unique())
    if len(valores) <= n_bins:  # pocos valores (p.ej. grado 1-7): un bin por valor
        return list((valores[:-1] + valores[1:]) / 2)
    cuantiles = np.unique(np.quantile(x.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))
    return list(cuantiles)


def _asignar_bin(x: pd.Series, cortes: list[float]) -> pd.Series:
    bins = pd.Series(np.searchsorted(np.asarray(cortes), x.to_numpy(dtype=float), side="left"), index=x.index)
    return bins.where(x.notna(), -1)  # -1 = nulo (bin propio)


def _tasa_por_bin(bins: pd.Series, y: pd.Series) -> pd.Series:
    datos = pd.DataFrame({"bin": bins, "y": y})
    return datos[datos["bin"] >= 0].groupby("bin")["y"].mean().sort_index()


def _forzar_monotonia(x: pd.Series, y: pd.Series, cortes: list[float], direccion: int) -> list[float]:
    """Fusiona bins vecinos hasta que la tasa de impago sea monotona en la
    direccion pedida (+1 creciente, -1 decreciente). Practica estandar en
    scorecards: evita que el modelo diga "a mas deuda, menos riesgo" por ruido."""
    cortes = list(cortes)
    while cortes:
        tasas = _tasa_por_bin(_asignar_bin(x, cortes), y).to_numpy()
        violaciones = np.where(np.diff(tasas) * direccion < 0)[0]
        if len(violaciones) == 0:
            break
        del cortes[violaciones[0]]
    return cortes


def _etiqueta_bin(valor, cortes: list[float] | None) -> str:
    """Texto legible del bin: '<= 0.12', '(0.12, 0.25]', '> 0.6', 'Nulo'."""
    if cortes is None:
        return str(valor)
    k = int(valor)
    if k == -1:
        return "Nulo"
    if not cortes:
        return "Todos"
    if k == 0:
        return f"<= {cortes[0]:.4g}"
    if k >= len(cortes):
        return f"> {cortes[-1]:.4g}"
    return f"({cortes[k - 1]:.4g}, {cortes[k]:.4g}]"


class BinningWoE:
    """Agrupa cada variable en bins y la reemplaza por su Weight of Evidence.

    Convencion: WoE = ln(%impagos_del_bin / %pagos_del_bin). WoE > 0 = bin
    mas riesgoso que el promedio. IV = suma((%impagos - %pagos) * WoE).
    Ajustar SOLO con datos de entrenamiento (fit) y aplicar (transform) al
    resto, para no filtrar informacion."""

    def __init__(self, n_bins: int = 10, monotonia: dict | None = None, suavizado: float = 0.5, min_categoria: float = 0.01):
        self.n_bins = n_bins
        self.monotonia = monotonia or {}
        self.suavizado = suavizado
        self.min_categoria = min_categoria
        self.cortes: dict[str, list[float]] = {}
        self.categorias: dict[str, set] = {}
        self.woe: dict[str, dict] = {}
        self.iv: dict[str, float] = {}
        self.tablas: dict[str, pd.DataFrame] = {}

    def _bins(self, nombre: str, x: pd.Series) -> pd.Series:
        if nombre in self.cortes:
            return _asignar_bin(x, self.cortes[nombre])
        conocidas = self.categorias[nombre]
        return x.astype(object).where(x.isin(conocidas), "OTRAS").fillna("NULO")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinningWoE":
        y = pd.Series(y).reset_index(drop=True)
        total_malos, total_buenos = y.sum(), (1 - y).sum()
        for nombre in X.columns:
            x = X[nombre].reset_index(drop=True)
            if _es_numerica(x):
                cortes = _cortes_iniciales(x, self.n_bins)
                if self.monotonia.get(nombre):
                    cortes = _forzar_monotonia(x, y, cortes, self.monotonia[nombre])
                self.cortes[nombre] = cortes
            else:
                frecuencia = x.value_counts(normalize=True)
                self.categorias[nombre] = set(frecuencia[frecuencia >= self.min_categoria].index)
            bins = self._bins(nombre, x)
            tabla = pd.DataFrame({"bin": bins, "y": y}).groupby("bin")["y"].agg(malos="sum", total="count")
            tabla["buenos"] = tabla["total"] - tabla["malos"]
            k = len(tabla)
            p_malos = (tabla["malos"] + self.suavizado) / (total_malos + self.suavizado * k)
            p_buenos = (tabla["buenos"] + self.suavizado) / (total_buenos + self.suavizado * k)
            tabla["woe"] = np.log(p_malos / p_buenos)
            tabla["tasa_impago"] = tabla["malos"] / tabla["total"]
            self.iv[nombre] = float(((p_malos - p_buenos) * tabla["woe"]).sum())
            self.woe[nombre] = tabla["woe"].to_dict()
            tabla = tabla.reset_index()
            tabla["rango"] = tabla["bin"].map(lambda b: _etiqueta_bin(b, self.cortes.get(nombre)))
            self.tablas[nombre] = tabla
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        salida = {}
        for nombre in self.woe:
            bins = self._bins(nombre, X[nombre].reset_index(drop=True))
            # Bin no visto en entrenamiento (p.ej. nulos que no hubo) -> WoE 0 = neutro
            salida[nombre] = bins.map(self.woe[nombre]).fillna(0.0).astype(float)
        return pd.DataFrame(salida)


def fuerza_iv(iv: float) -> str:
    if iv < 0.02:
        return "sin poder"
    if iv < 0.1:
        return "debil"
    if iv < 0.3:
        return "media"
    if iv <= 0.5:
        return "fuerte"
    return "sospechosa (revisar fuga)"


# ---------------------------------------------------------------------------
# 3) Seleccion de variables
# ---------------------------------------------------------------------------
def _vif(X: pd.DataFrame) -> dict:
    """Variance Inflation Factor: que tanto se explica una variable con las
    demas (>5-10 = redundante)."""
    datos = X.apply(lambda c: c.fillna(c.median())).astype(float)
    datos = (datos - datos.mean()) / datos.std(ddof=0).replace(0, 1)
    resultado = {}
    for columna in datos.columns:
        otras = datos.drop(columns=columna).to_numpy()
        objetivo = datos[columna].to_numpy()
        if otras.shape[1] == 0:
            resultado[columna] = 1.0
            continue
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(otras)), otras]), objetivo, rcond=None)
        prediccion = np.column_stack([np.ones(len(otras)), otras]) @ coef
        r2 = 1 - ((objetivo - prediccion) ** 2).sum() / ((objetivo - objetivo.mean()) ** 2).sum()
        resultado[columna] = float(1 / max(1 - r2, 1e-6))
    return resultado


def seleccionar_variables(df: pd.DataFrame, contrato: dict) -> tuple[list[str], pd.DataFrame]:
    """Aplica las reglas del contrato y regresa (variables_elegidas, reporte).
    El reporte trae una fila por candidata con su IV y la razon de la decision."""
    reglas = contrato["seleccion"]
    especificacion = contrato["features"]
    candidatas = [c for c in df.columns if c != TARGET and c in especificacion]
    monotonia = {n: especificacion[n].get("monotonia", 0) for n in candidatas}

    binning = BinningWoE(n_bins=reglas["bins"], monotonia=monotonia).fit(df[candidatas], df[TARGET])
    filas = []
    for nombre in candidatas:
        iv = binning.iv[nombre]
        fila = {"variable": nombre, "iv": round(iv, 4), "fuerza": fuerza_iv(iv), "decision": "seleccionada", "razon": ""}
        if especificacion[nombre].get("uso") == "excluida":
            fila.update(decision="excluida", razon=especificacion[nombre].get("razon", "excluida por contrato"))
        elif iv < reglas["iv_minimo"]:
            fila.update(decision="excluida", razon=f"IV {iv:.3f} < {reglas['iv_minimo']}: sin poder predictivo")
        elif iv > reglas["iv_sospechoso"]:
            fila["razon"] = f"IV {iv:.2f} muy alto: posible fuga; se conserva porque es un dato disponible al otorgar el credito (ver contrato)"
        filas.append(fila)
    reporte = pd.DataFrame(filas)

    # Redundancia: entre dos variables numericas muy correlacionadas, se queda la de mayor IV
    elegidas = reporte.loc[reporte["decision"] == "seleccionada", "variable"].tolist()
    numericas = [v for v in elegidas if _es_numerica(df[v])]
    correlacion = df[numericas].corr(method="spearman").abs()
    for i, a in enumerate(numericas):
        for b in numericas[i + 1:]:
            if a not in elegidas or b not in elegidas or correlacion.loc[a, b] <= reglas["correlacion_maxima"]:
                continue
            debil, fuerte = (a, b) if binning.iv[a] < binning.iv[b] else (b, a)
            elegidas.remove(debil)
            reporte.loc[reporte["variable"] == debil, ["decision", "razon"]] = [
                "excluida", f"redundante con {fuerte} (|r| = {correlacion.loc[a, b]:.2f}, menor IV)"
            ]

    vif = _vif(df[[v for v in elegidas if _es_numerica(df[v])]])
    reporte["vif"] = reporte["variable"].map(vif).round(2)
    reporte = reporte.sort_values(["decision", "iv"], ascending=[False, False]).reset_index(drop=True)
    return elegidas, reporte


# ---------------------------------------------------------------------------
# 4) Particion y 5) validacion
# ---------------------------------------------------------------------------
def dividir(df: pd.DataFrame, contrato: dict) -> pd.Series:
    """Etiqueta cada fila como entrenamiento / validacion / prueba,
    estratificado por el target y con semilla fija (reproducible)."""
    config = contrato["particion"]
    rng = np.random.default_rng(config["semilla"])
    etiquetas = pd.Series("entrenamiento", index=df.index)
    for _, grupo in df.groupby(TARGET):
        indices = rng.permutation(grupo.index.to_numpy())
        n_prueba = int(round(len(indices) * config["prueba"]))
        n_validacion = int(round(len(indices) * config["validacion"]))
        etiquetas.loc[indices[:n_prueba]] = "prueba"
        etiquetas.loc[indices[n_prueba:n_prueba + n_validacion]] = "validacion"
    return etiquetas


def validar_features(df: pd.DataFrame, contrato: dict, columnas: list[str]) -> dict:
    """Rangos y nulos de cada feature del modelo contra el contrato."""
    chequeos = []
    for nombre in columnas:
        especificacion = contrato["features"][nombre]
        serie = df[nombre]
        if not especificacion.get("nulos_permitidos"):
            nulos = int(serie.isna().sum())
            chequeos.append({"chequeo": f"sin_nulos:{nombre}", "exito": nulos == 0, "fallas": nulos})
        if "rango_valido" in especificacion and _es_numerica(serie):
            minimo, maximo = especificacion["rango_valido"]
            fuera = int(((serie < minimo) | (serie > maximo)).sum())
            chequeos.append({"chequeo": f"rango:{nombre}", "exito": fuera == 0, "fallas": fuera})
    return {
        "exito_global": all(c["exito"] for c in chequeos),
        "chequeos_exitosos": sum(c["exito"] for c in chequeos),
        "chequeos_totales": len(chequeos),
        "detalle": chequeos,
    }


def huella_dataset(df: pd.DataFrame) -> str:
    """Hash del contenido: identifica la version exacta del dataset."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def preparar_dataset_entrenamiento(df_credit: pd.DataFrame, contrato: dict) -> dict:
    """Pipeline completo de la Semana 14 sobre credit_risk (Silver)."""
    candidatas = features_entrenamiento(df_credit, contrato)
    elegidas, reporte = seleccionar_variables(candidatas, contrato)
    dataset = candidatas.copy()
    dataset["particion_modelo"] = dividir(dataset, contrato)

    columnas_modelo = features_modelo(contrato)
    calidad = validar_features(dataset, contrato, columnas_modelo)
    distribucion = dataset["grado_credito"].map(dict(enumerate(GRADOS, start=1))).value_counts(normalize=True)

    metadata = {
        "fecha": None,  # se llena al publicar (construir_dataset_entrenamiento)
        "version_dataset": huella_dataset(dataset),
        "version_contrato": contrato.get("version_contrato"),
        "filas": int(len(dataset)),
        "tasa_impago": round(float(dataset[TARGET].mean()), 4),
        "filas_por_particion": dataset["particion_modelo"].value_counts().to_dict(),
        "tasa_impago_por_particion": dataset.groupby("particion_modelo")[TARGET].mean().round(4).to_dict(),
        "features_contrato": columnas_modelo,
        "features_seleccionadas": elegidas,
        "features_excluidas": reporte.loc[reporte["decision"] == "excluida", ["variable", "razon"]].to_dict(orient="records"),
        "distribucion_grados": {g: round(float(distribucion.get(g, 0.0)), 4) for g in GRADOS},
        # Percentiles del entrenamiento para alinear la cartera (ver alinear_por_cuantiles)
        "cuantiles_referencia": cuantiles_referencia(
            dataset[dataset["particion_modelo"] == "entrenamiento"],
            [c for c in columnas_modelo if contrato["features"][c]["tipo"] == "numerica"],
        ),
        "calidad": calidad,
    }
    return {"dataset": dataset, "reporte_seleccion": reporte, "metadata": metadata}


# ---------------------------------------------------------------------------
# E/S con MinIO
# ---------------------------------------------------------------------------
_PATRON_PARTICION = re.compile(r"/(\d{4})/(\d{2})/(\d{2})/valid/")


def particion_mas_reciente(fs, raiz: str, fecha_max: str) -> str | None:
    """La particion Silver mas reciente (YYYY/MM/DD) de `raiz` que no sea
    posterior a `fecha_max`, o None si no hay ninguna.

    Por que: el entrenamiento es semanal y su fecha logica puede no tener
    particion (p.ej. el dataset se descargo despues). Exigir la fecha exacta
    hacia que la semana completa se saltara; el dataset etiquetado casi no
    cambia, asi que usar la ultima version disponible es lo correcto. La
    particion usada queda registrada en la metadata (linaje)."""
    if not fs.exists(raiz):
        return None
    limite = fecha_max.replace("-", "/")
    candidatas = set()
    for ruta in fs.find(raiz):
        coincidencia = _PATRON_PARTICION.search("/" + ruta.lstrip("/"))
        if coincidencia:
            particion = "/".join(coincidencia.groups())
            if particion <= limite:
                candidatas.add(particion)
    return max(candidatas) if candidatas else None


def construir_dataset_entrenamiento(fecha: str, fecha_max: str | None = None) -> dict:
    """Lee credit_risk de Silver (la particion mas reciente hasta `fecha`),
    prepara el dataset y lo publica en Gold bajo `fecha`:
    gold_features_entrenamiento, reporte_seleccion_variables y metadata."""
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET, SILVER_BUCKET, STORAGE_OPTIONS

    gold = os.getenv("GOLD_BUCKET", "gold-layer")
    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    tope = fecha_max or fecha
    particion = particion_mas_reciente(fs, f"{SILVER_BUCKET}/credit_risk", tope)
    if particion is None:
        raise FileNotFoundError(f"No hay ninguna particion de credit_risk en Silver hasta {tope}")
    df_credit = pd.read_parquet(f"s3://{SILVER_BUCKET}/credit_risk/{particion}/valid", storage_options=STORAGE_OPTIONS)
    resultado = preparar_dataset_entrenamiento(df_credit, cargar_contrato())
    if not resultado["metadata"]["calidad"]["exito_global"]:
        fallidos = [c for c in resultado["metadata"]["calidad"]["detalle"] if not c["exito"]]
        raise ValueError(f"Dataset de entrenamiento con errores de calidad: {fallidos}")

    resultado["dataset"].to_parquet(
        f"s3://{gold}/gold_features_entrenamiento/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS, index=False
    )
    resultado["reporte_seleccion"].to_parquet(
        f"s3://{gold}/reporte_seleccion_variables/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS, index=False
    )
    resultado["metadata"]["fecha"] = fecha
    resultado["metadata"]["particion_origen"] = particion  # linaje: que Silver se uso
    metadata = {**resultado["metadata"], "generado_en": datetime.now(timezone.utc).isoformat()}
    with fs.open(f"{gold}/_features_metadata/{fecha.replace('/', '-')}.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(construir_dataset_entrenamiento(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
