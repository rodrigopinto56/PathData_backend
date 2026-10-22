"""
scripts/modelo_credito.py

Semana 15: modelo predictivo de riesgo crediticio (reemplaza al modelo
de la Semana 11, ver docs/08_modelo_riesgo.md).

Ciclo completo, de entrenamiento a calificacion:
  1) Dos candidatos sobre el dataset de la Semana 14:
       - scorecard: regresion logistica sobre WoE (explicable en puntos)
       - boosting:  HistGradientBoosting con restricciones de monotonia
  2) Evaluacion con validacion cruzada estratificada (AUC, Gini, KS, Brier)
     y prueba final en un conjunto que nunca se usa para ajustar nada.
  3) Calibracion: Platt sobre validacion + ajuste a la tasa de impago
     objetivo de la cartera (el dataset de entrenamiento tiene ~22%).
  4) Pruebas de sentido de negocio (monotonia) y auditoria de equidad.
  5) Registro versionado + regla campeon/retador: un modelo nuevo solo
     reemplaza al vigente si no lo empeora y pasa las pruebas.
  6) Calificacion de la cartera: PD calibrada, nivel de riesgo, puntos de
     scorecard y las 3 razones principales de cada cliente.

Configuracion: config/modelo_credito.yml. Features: config/features.yml.
Todo es pandas/scikit-learn puro salvo la clase RegistroModelos (que
recibe un sistema de archivos fsspec: MinIO en produccion, disco en pruebas)
y las funciones entrenar_y_registrar / cargar_campeon_minio.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from features import (
    GRADOS,
    particion_mas_reciente,
    TARGET,
    BinningWoE,
    alinear_por_cuantiles,
    cargar_contrato,
    features_cartera,
    psi,
)

RUTAS_CONFIG = [
    "/opt/airflow/config/modelo_credito.yml",
    os.path.join(os.path.dirname(__file__), "..", "config", "modelo_credito.yml"),
]

# Como se explica cada variable cuando EMPEORA el riesgo de un cliente
RAZONES = {
    "ratio_prestamo_ingreso": "Préstamo alto respecto a su ingreso",
    "grado_credito": "Historial crediticio débil",
    "ingreso_anual_log": "Ingreso bajo",
    "monto_prestamo_log": "Monto del préstamo",
    "tasa_interes": "Tasa de interés alta",
    "proposito": "Propósito del préstamo de mayor riesgo",
}


def cargar_config(ruta: str | None = None) -> dict:
    for candidata in ([ruta] if ruta else RUTAS_CONFIG):
        if candidata and os.path.exists(candidata):
            with open(candidata) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No se encontro config/modelo_credito.yml en {RUTAS_CONFIG}")


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------
class ModeloScorecard:
    """WoE + regresion logistica. Cada variable aporta puntos: el cliente
    "suma" o "resta" riesgo de forma explicable."""

    nombre = "scorecard"

    def __init__(self, features: list[str], monotonia: dict, C: float = 1.0):
        self.features = features
        self.monotonia = monotonia
        self.C = C

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModeloScorecard":
        self.binning = BinningWoE(monotonia=self.monotonia).fit(X[self.features], y)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(self.binning.transform(X[self.features]), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self.binning.transform(X[self.features]))[:, 1]

    def contribuciones(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aporte de cada variable a los log-odds de impago (positivo = sube el riesgo)."""
        woe = self.binning.transform(X[self.features])
        return woe * self.lr.coef_[0]

    def tabla_puntos(self, puntos_cfg: dict) -> pd.DataFrame:
        """Puntos por bin, convencion clasica (mas puntos = menos riesgo)."""
        factor = puntos_cfg["pdo"] / np.log(2)
        filas = []
        for coef, nombre in zip(self.lr.coef_[0], self.features):
            for _, fila in self.binning.tablas[nombre].iterrows():
                filas.append({
                    "variable": nombre,
                    "rango": fila["rango"],
                    "clientes": int(fila["total"]),
                    "tasa_impago": round(float(fila["tasa_impago"]), 4),
                    "woe": round(float(fila["woe"]), 4),
                    "puntos": round(float(-coef * fila["woe"] * factor), 1),
                })
        return pd.DataFrame(filas)


class ModeloBoosting:
    """Retador no lineal. Las restricciones de monotonia garantizan por
    construccion que, p.ej., a mayor ratio prestamo/ingreso nunca baje el riesgo."""

    nombre = "boosting"

    def __init__(self, features: list[str], monotonia: dict, **parametros):
        self.features = features
        self.monotonia = monotonia
        self.parametros = parametros

    def _matriz(self, X: pd.DataFrame) -> pd.DataFrame:
        datos = X[self.features].copy()
        for columna in self.categoricas:
            codigos = {c: i for i, c in enumerate(self.categorias[columna])}
            datos[columna] = datos[columna].map(codigos).astype(float)  # no vista -> NaN
        return datos.astype(float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModeloBoosting":
        self.categoricas = [c for c in self.features if not pd.api.types.is_numeric_dtype(X[c])]
        self.categorias = {c: sorted(X[c].dropna().unique()) for c in self.categoricas}
        self.modelo = HistGradientBoostingClassifier(
            categorical_features=[self.features.index(c) for c in self.categoricas] or None,
            monotonic_cst=[0 if c in self.categoricas else int(self.monotonia.get(c, 0)) for c in self.features],
            random_state=42,
            **self.parametros,
        ).fit(self._matriz(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self._matriz(X))[:, 1]


def _copia_sin_entrenar(plantilla):
    """Nueva instancia con la misma configuracion (para cada pliegue de la CV)."""
    if isinstance(plantilla, ModeloScorecard):
        return ModeloScorecard(plantilla.features, plantilla.monotonia, C=plantilla.C)
    return ModeloBoosting(plantilla.features, plantilla.monotonia, **plantilla.parametros)


def crear_candidatos(features: list[str], monotonia: dict, config: dict) -> list:
    c = config["candidatos"]
    return [
        ModeloScorecard(features, monotonia, C=c["scorecard"]["regularizacion_C"]),
        ModeloBoosting(
            features, monotonia,
            max_iter=c["boosting"]["max_iter"], learning_rate=c["boosting"]["learning_rate"],
            max_leaf_nodes=c["boosting"]["max_leaf_nodes"], min_samples_leaf=c["boosting"]["min_samples_leaf"],
        ),
    ]


# ---------------------------------------------------------------------------
# Metricas, calibracion y ajuste de tasa base
# ---------------------------------------------------------------------------
def metricas(y: pd.Series, p: np.ndarray) -> dict:
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(float(np.max(tpr - fpr)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "tasa_real": round(float(np.mean(y)), 4),
        "tasa_predicha": round(float(np.mean(p)), 4),
    }


def curva_calibracion(y: pd.Series, p: np.ndarray, bins: int = 10) -> list[dict]:
    datos = pd.DataFrame({"y": np.asarray(y), "p": p})
    datos["bin"] = pd.qcut(datos["p"].rank(method="first"), bins, labels=False)
    tabla = datos.groupby("bin").agg(pd_predicha=("p", "mean"), tasa_real=("y", "mean"), clientes=("y", "size"))
    return tabla.round(4).reset_index(drop=True).to_dict(orient="records")


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class CalibradorPlatt:
    """Ajusta una sigmoide sobre el logit de la probabilidad del modelo,
    usando el conjunto de validacion (nunca el de entrenamiento ni el de prueba)."""

    def fit(self, p: np.ndarray, y: pd.Series) -> "CalibradorPlatt":
        self.lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def ajustar_tasa_base(p: np.ndarray, tasa_origen: float, tasa_objetivo: float) -> np.ndarray:
    """Re-escala probabilidades calibradas a otra tasa de impago (correccion
    de prior): conserva el orden de riesgo, cambia el nivel. Necesario porque
    el entrenamiento tiene ~22% de impago y una cartera tipica ~5%."""
    odds = p / (1 - p) * (tasa_objetivo / (1 - tasa_objetivo)) / (tasa_origen / (1 - tasa_origen))
    return odds / (1 + odds)


def validacion_cruzada(candidato_fabrica, X: pd.DataFrame, y: pd.Series, pliegues: int, semilla: int) -> dict:
    resultados = []
    for idx_fit, idx_eval in StratifiedKFold(pliegues, shuffle=True, random_state=semilla).split(X, y):
        modelo = candidato_fabrica().fit(X.iloc[idx_fit], y.iloc[idx_fit])
        resultados.append(metricas(y.iloc[idx_eval], modelo.predict_proba(X.iloc[idx_eval])))
    tabla = pd.DataFrame(resultados)
    return {
        metrica: {"media": round(float(tabla[metrica].mean()), 4), "desviacion": round(float(tabla[metrica].std()), 4)}
        for metrica in ["auc", "gini", "ks", "brier"]
    }


# ---------------------------------------------------------------------------
# Artefacto que se registra y se usa para calificar
# ---------------------------------------------------------------------------
class ModeloRiesgoRegistrado:
    """Todo lo necesario para calificar la cartera igual que en el entrenamiento."""

    def __init__(self, modelo, scorecard: ModeloScorecard, calibrador: CalibradorPlatt, *,
                 tasa_origen: float, tasa_objetivo: float, umbral_alto: float,
                 cuantiles_referencia: dict, distribucion_grados: dict, puntos_cfg: dict, version: str):
        self.modelo = modelo
        self.scorecard = scorecard
        self.calibrador = calibrador
        self.tasa_origen = tasa_origen
        self.tasa_objetivo = tasa_objetivo
        self.umbral_alto = umbral_alto
        self.cuantiles_referencia = cuantiles_referencia
        self.distribucion_grados = distribucion_grados
        self.puntos_cfg = puntos_cfg
        self.version = version

    @property
    def features(self) -> list[str]:
        return self.modelo.features

    def pd_calibrada(self, X: pd.DataFrame) -> np.ndarray:
        """PD en la escala del entrenamiento (para evaluar contra datos etiquetados)."""
        return self.calibrador.transform(self.modelo.predict_proba(X))

    def pd_cartera(self, X: pd.DataFrame) -> np.ndarray:
        """PD en la escala de la cartera (tasa objetivo)."""
        return ajustar_tasa_base(self.pd_calibrada(X), self.tasa_origen, self.tasa_objetivo)

    def puntos(self, X: pd.DataFrame) -> np.ndarray:
        """Puntaje tipo buro: mas puntos = menos riesgo (sobre la PD de cartera)."""
        cfg = self.puntos_cfg
        factor = cfg["pdo"] / np.log(2)
        offset = cfg["puntos_base"] - factor * np.log(cfg["odds_base"])
        p = np.clip(self.pd_cartera(X), 1e-6, 1 - 1e-6)
        return np.round(offset + factor * np.log((1 - p) / p))

    def razones(self, X: pd.DataFrame, top: int = 3) -> list[str]:
        """Las variables que mas EMPUJAN el riesgo de cada cliente hacia arriba
        (aporte positivo en la scorecard), en lenguaje de negocio."""
        aportes = self.scorecard.contribuciones(X)
        salida = []
        for _, fila in aportes.iterrows():
            positivas = fila[fila > 0.05].sort_values(ascending=False).head(top)
            salida.append("; ".join(RAZONES.get(v, v) for v in positivas.index) or "Sin factores de riesgo relevantes")
        return salida

    def calificar_cartera(self, df_pf: pd.DataFrame, contrato: dict) -> pd.DataFrame:
        """personal_finance_ml (Silver) -> mismo df + riesgo_score, riesgo_flag,
        puntaje_riesgo, razones_riesgo, grado_credito_proxy.
        Solo se califica a quien tiene prestamo (la PD es de un credito);
        el resto queda como 'Sin préstamo' (y PD nula)."""
        crudas = features_cartera(df_pf, contrato, self.distribucion_grados)
        con_prestamo = crudas["tiene_prestamo"] == 1
        alineadas = alinear_por_cuantiles(crudas, self.cuantiles_referencia, con_prestamo)

        resultado = df_pf.reset_index(drop=True).copy()
        resultado["riesgo_score"] = np.nan
        resultado["puntaje_riesgo"] = np.nan
        resultado["riesgo_flag"] = "Sin préstamo"
        resultado["razones_riesgo"] = ""
        resultado["grado_credito_proxy"] = crudas["grado_credito"].map(dict(enumerate(GRADOS, start=1)))
        if con_prestamo.any():
            X = alineadas.loc[con_prestamo]
            pd_cartera = self.pd_cartera(X)
            resultado.loc[con_prestamo, "riesgo_score"] = np.round(pd_cartera, 4)
            resultado.loc[con_prestamo, "puntaje_riesgo"] = self.puntos(X)
            resultado.loc[con_prestamo, "riesgo_flag"] = np.where(pd_cartera >= self.umbral_alto, "Alto", "Bajo")
            resultado.loc[con_prestamo, "razones_riesgo"] = self.razones(X)
        return resultado


# ---------------------------------------------------------------------------
# Pruebas de negocio y equidad
# ---------------------------------------------------------------------------
def pruebas_monotonia(artefacto: ModeloRiesgoRegistrado, X_ref: pd.DataFrame, reglas: dict) -> dict:
    """Perfil tipico (mediana / moda) y se mueve UNA variable a la vez: la PD
    debe subir (o mantenerse) en la direccion que dicta el negocio."""
    perfil = {}
    for c in artefacto.features:
        perfil[c] = X_ref[c].median() if pd.api.types.is_numeric_dtype(X_ref[c]) else X_ref[c].mode().iloc[0]
    resultados = {}
    for variable, direccion in reglas.items():
        if variable not in artefacto.features:
            continue
        if variable == "grado_credito":
            valores = np.arange(1, 8, dtype=float)
        else:
            valores = np.nanquantile(X_ref[variable].astype(float), np.linspace(0.05, 0.95, 10))
        escenarios = pd.DataFrame([{**perfil, variable: v} for v in valores])
        p = artefacto.pd_calibrada(escenarios)
        exito = bool(np.all(np.diff(p) * direccion >= -1e-9))
        resultados[variable] = {"exito": exito, "pd": [round(float(x), 4) for x in p]}
    return resultados


def auditoria_equidad(grupos: pd.Series, es_alto: pd.Series, ratio_minimo: float) -> dict:
    """Regla de los 4/5 sobre la tasa de 'aprobacion' (no riesgo alto)."""
    datos = pd.DataFrame({"grupo": grupos, "alto": es_alto.astype(float)}).dropna()
    tabla = datos.groupby("grupo")["alto"].agg(["mean", "size"])
    aprobacion = 1 - tabla["mean"]
    ratio = float(aprobacion.min() / aprobacion.max()) if aprobacion.max() > 0 else float("nan")
    return {
        "por_grupo": {str(g): {"pct_riesgo_alto": round(float(f["mean"]) * 100, 2), "clientes": int(f["size"])}
                      for g, f in tabla.iterrows()},
        "ratio_aprobacion": round(ratio, 3),
        "cumple_regla_4_5": bool(ratio >= ratio_minimo),
    }


def _banda_edad(edad: pd.Series) -> pd.Series:
    return pd.cut(edad, [0, 29, 44, 200], labels=["18-29", "30-44", "45+"]).astype(str)


# ---------------------------------------------------------------------------
# Entrenamiento completo (puro: sin E/S)
# ---------------------------------------------------------------------------
def entrenar_modelo(dataset: pd.DataFrame, metadata_dataset: dict, config: dict, contrato: dict,
                    campeon_actual: ModeloRiesgoRegistrado | None = None,
                    df_cartera: pd.DataFrame | None = None, version: str | None = None) -> dict:
    version = version or datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
    features = metadata_dataset["features_seleccionadas"]
    monotonia = {f: contrato["features"][f].get("monotonia", 0) for f in features}
    entrenamiento = dataset[dataset["particion_modelo"] == "entrenamiento"].reset_index(drop=True)
    validacion = dataset[dataset["particion_modelo"] == "validacion"].reset_index(drop=True)
    prueba = dataset[dataset["particion_modelo"] == "prueba"].reset_index(drop=True)
    y_tr, y_val, y_te = entrenamiento[TARGET], validacion[TARGET], prueba[TARGET]

    cv_cfg = config["validacion_cruzada"]
    tasa_origen = float(y_tr.mean())
    tasa_objetivo = config["calibracion"]["tasa_impago_objetivo"]
    umbral_alto = config["clasificacion"]["multiplicador_riesgo_alto"] * tasa_objetivo

    candidatos, artefactos = {}, {}
    scorecard = None
    for plantilla in crear_candidatos(features, monotonia, config):
        cv = validacion_cruzada(
            lambda p=plantilla: _copia_sin_entrenar(p), entrenamiento, y_tr, cv_cfg["pliegues"], cv_cfg["semilla"]
        )
        modelo = _copia_sin_entrenar(plantilla).fit(entrenamiento, y_tr)
        if isinstance(modelo, ModeloScorecard):
            scorecard = modelo
        calibrador = CalibradorPlatt().fit(modelo.predict_proba(validacion), y_val)
        p_sin_calibrar = modelo.predict_proba(prueba)
        p_prueba = calibrador.transform(p_sin_calibrar)
        candidatos[modelo.nombre] = {
            "descripcion": config["candidatos"][modelo.nombre]["descripcion"],
            "validacion_cruzada": cv,
            "prueba_sin_calibrar": metricas(y_te, p_sin_calibrar),
            "prueba": metricas(y_te, p_prueba),
            "curva_calibracion": curva_calibracion(y_te, p_prueba),
        }
        artefactos[modelo.nombre] = (modelo, calibrador)

    # --- Scorecard vs retador ---
    reglas = config["seleccion_campeon"]
    sc, bo = candidatos["scorecard"]["prueba"], candidatos["boosting"]["prueba"]
    if bo["auc"] - sc["auc"] >= reglas["ventaja_minima_auc_retador"] and bo["brier"] <= sc["brier"]:
        elegido, razon = "boosting", f"mejora el AUC en {bo['auc'] - sc['auc']:.3f} sin empeorar la calibracion"
    else:
        elegido, razon = "scorecard", (
            f"el retador no supera el margen de {reglas['ventaja_minima_auc_retador']} de AUC "
            f"({bo['auc'] - sc['auc']:+.3f}); se prefiere el modelo explicable"
        )

    modelo, calibrador = artefactos[elegido]
    artefacto = ModeloRiesgoRegistrado(
        modelo, scorecard, calibrador,
        tasa_origen=tasa_origen, tasa_objetivo=tasa_objetivo, umbral_alto=umbral_alto,
        cuantiles_referencia={k: v for k, v in metadata_dataset["cuantiles_referencia"].items() if k in features},
        distribucion_grados=metadata_dataset["distribucion_grados"],
        puntos_cfg=config["scorecard_puntos"], version=version,
    )

    # --- Pruebas de negocio y equidad (entrenamiento) ---
    monotonia_resultado = pruebas_monotonia(artefacto, entrenamiento, config["pruebas_negocio"]["monotonia"])
    p_te = artefacto.pd_calibrada(prueba)
    umbral_origen = config["clasificacion"]["multiplicador_riesgo_alto"] * tasa_origen
    equidad = {
        "edad_en_prueba": auditoria_equidad(
            _banda_edad(prueba["edad"]), pd.Series(p_te >= min(umbral_origen, 0.99)), config["equidad"]["ratio_minimo_aprobacion"]
        )
    }

    # --- Cartera: cambio de distribucion y equidad por genero ---
    evaluacion_cartera = None
    if df_cartera is not None:
        crudas = features_cartera(df_cartera, contrato, artefacto.distribucion_grados)
        con_prestamo = crudas["tiene_prestamo"] == 1
        calificada = artefacto.calificar_cartera(df_cartera, contrato)
        evaluacion_cartera = {
            "clientes_con_prestamo": int(con_prestamo.sum()),
            "psi_sin_alinear": {
                f: round(psi(entrenamiento[f], crudas.loc[con_prestamo, f]), 3)
                for f in features if pd.api.types.is_numeric_dtype(entrenamiento[f])
            },
            "pd_promedio": round(float(calificada["riesgo_score"].mean()), 4),
            "pct_riesgo_alto": round(float((calificada["riesgo_flag"] == "Alto").sum() / max(con_prestamo.sum(), 1)) * 100, 2),
            "equidad_genero": auditoria_equidad(
                crudas.loc[con_prestamo, "genero"], (calificada.loc[con_prestamo, "riesgo_flag"] == "Alto"),
                config["equidad"]["ratio_minimo_aprobacion"],
            ) if "genero" in crudas and crudas["genero"].notna().any() else None,
        }
        equidad["genero_en_cartera"] = (evaluacion_cartera or {}).get("equidad_genero")

    # --- Campeon vigente vs modelo nuevo ---
    pasa_pruebas = all(r["exito"] for r in monotonia_resultado.values())
    promover, razon_promocion = pasa_pruebas, "primer modelo registrado"
    auc_nuevo = metricas(y_te, p_te)["auc"]
    if not pasa_pruebas:
        razon_promocion = "no pasa las pruebas de sentido de negocio (monotonia)"
    elif campeon_actual is not None:
        if campeon_actual.features != artefacto.features:
            razon_promocion = "el contrato de features cambio: se reemplaza el campeon"
        else:
            auc_campeon = metricas(y_te, campeon_actual.pd_calibrada(prueba))["auc"]
            promover = auc_nuevo >= auc_campeon - reglas["tolerancia_auc_vs_campeon"]
            razon_promocion = (
                f"AUC nuevo {auc_nuevo:.4f} vs campeon {campeon_actual.version} {auc_campeon:.4f}: "
                + ("se promueve" if promover else "se conserva el campeon")
            )

    card = {
        "version": version,
        "entrenado_en": datetime.now(timezone.utc).isoformat(),
        "proposito": "Estimar la probabilidad de impago (PD) de clientes con prestamo",
        "dataset": {k: metadata_dataset.get(k) for k in ["version_dataset", "filas", "tasa_impago", "filas_por_particion", "fecha"]},
        "features": features,
        "features_excluidas": metadata_dataset.get("features_excluidas", []),
        "candidatos": candidatos,
        "modelo_elegido": elegido,
        "razon_eleccion": razon,
        "metricas_prueba": metricas(y_te, p_te),
        "calibracion": {"metodo": "platt", "tasa_entrenamiento": round(tasa_origen, 4), "tasa_objetivo_cartera": tasa_objetivo},
        "umbral_riesgo_alto": round(umbral_alto, 4),
        "escala_indice_pd": [round(tasa_objetivo * 0.5, 4), round(umbral_alto, 4)],
        "coeficientes_scorecard": {f: round(float(c), 4) for f, c in zip(scorecard.features, scorecard.lr.coef_[0])},
        "pruebas_monotonia": monotonia_resultado,
        "equidad": equidad,
        "evaluacion_cartera": evaluacion_cartera,
        "promovido": promover,
        "razon_promocion": razon_promocion,
        "limitaciones": [
            "Se entrena con un dataset publico (credit_risk) distinto a la cartera: la cartera no tiene etiqueta de impago, por eso no se puede medir el AUC sobre ella.",
            "Las features de la cartera se alinean por cuantiles: el riesgo es relativo a la propia cartera (ver psi_sin_alinear).",
            "El grado de credito de la cartera es un proxy por cuantiles del credit_score.",
            "La PD se re-escala a una tasa objetivo configurable (tasa_impago_objetivo); no es una tasa observada.",
        ],
    }
    return {
        "artefacto": artefacto,
        "card": card,
        "tabla_puntos": scorecard.tabla_puntos(config["scorecard_puntos"]),
        "promover": promover,
    }


# ---------------------------------------------------------------------------
# Registro de modelos (fsspec: MinIO en produccion, disco en pruebas)
# ---------------------------------------------------------------------------
class RegistroModelos:
    def __init__(self, fs, raiz: str):
        self.fs = fs
        self.raiz = raiz.rstrip("/")

    def _ruta(self, *partes: str) -> str:
        return "/".join([self.raiz, *partes])

    def guardar(self, version: str, artefacto, card: dict, tabla_puntos: pd.DataFrame | None = None) -> None:
        self.fs.makedirs(self._ruta(version), exist_ok=True)
        buffer = io.BytesIO()
        joblib.dump(artefacto, buffer)
        with self.fs.open(self._ruta(version, "modelo.joblib"), "wb") as f:
            f.write(buffer.getvalue())
        with self.fs.open(self._ruta(version, "model_card.json"), "w") as f:
            json.dump(card, f, indent=2, ensure_ascii=False, default=str)
        if tabla_puntos is not None:
            with self.fs.open(self._ruta(version, "tabla_puntos.csv"), "w") as f:
                tabla_puntos.to_csv(f, index=False)

    def promover(self, version: str, razon: str) -> None:
        with self.fs.open(self._ruta("campeon.json"), "w") as f:
            json.dump({"version": version, "promovido_en": datetime.now(timezone.utc).isoformat(), "razon": razon}, f, indent=2)

    def version_campeon(self) -> str | None:
        ruta = self._ruta("campeon.json")
        if not self.fs.exists(ruta):
            return None
        with self.fs.open(ruta, "r") as f:
            return json.load(f)["version"]

    def cargar(self, version: str) -> ModeloRiesgoRegistrado:
        # Solo se cargan artefactos que escribio este mismo pipeline (joblib
        # ejecuta codigo al deserializar: el bucket no debe ser escribible por terceros).
        with self.fs.open(self._ruta(version, "modelo.joblib"), "rb") as f:
            return joblib.load(io.BytesIO(f.read()))

    def card(self, version: str) -> dict:
        with self.fs.open(self._ruta(version, "model_card.json"), "r") as f:
            return json.load(f)

    def campeon(self) -> tuple[ModeloRiesgoRegistrado | None, dict | None]:
        version = self.version_campeon()
        if version is None:
            return None, None
        return self.cargar(version), self.card(version)


# ---------------------------------------------------------------------------
# E/S con MinIO
# ---------------------------------------------------------------------------
def _registro_minio() -> RegistroModelos:
    import s3fs

    from modelo_riesgo import MINIO_ENDPOINT, MINIO_KEY, MINIO_SECRET

    config = cargar_config()
    bucket = os.getenv("ML_BUCKET", config["registro"]["bucket"])
    fs = s3fs.S3FileSystem(key=MINIO_KEY, secret=MINIO_SECRET, client_kwargs={"endpoint_url": MINIO_ENDPOINT})
    if not fs.exists(bucket):
        fs.mkdir(bucket)
    return RegistroModelos(fs, f"{bucket}/{config['registro']['ruta']}")


def cargar_campeon_minio() -> tuple[ModeloRiesgoRegistrado | None, dict | None]:
    """Lo usa construir_gold: si no hay campeon todavia, regresa (None, None)
    y Gold sigue con el modelo de la Semana 11."""
    try:
        return _registro_minio().campeon()
    except FileNotFoundError:
        return None, None


def entrenar_y_registrar(fecha: str, fecha_max: str | None = None) -> dict:
    """Lee el dataset de la Semana 14 de gold-layer, entrena, registra la
    version y la promueve a campeon si corresponde."""
    from modelo_riesgo import SILVER_BUCKET, STORAGE_OPTIONS

    gold = os.getenv("GOLD_BUCKET", "gold-layer")
    dataset = pd.read_parquet(f"s3://{gold}/gold_features_entrenamiento/{fecha}/data.parquet", storage_options=STORAGE_OPTIONS)
    registro = _registro_minio()
    with registro.fs.open(f"{gold}/_features_metadata/{fecha.replace('/', '-')}.json", "r") as f:
        metadata_dataset = json.load(f)
    # Cartera para medir PSI y equidad: la particion mas reciente disponible
    particion_cartera = particion_mas_reciente(registro.fs, f"{SILVER_BUCKET}/personal_finance_ml", fecha_max or fecha)
    df_cartera = (
        pd.read_parquet(f"s3://{SILVER_BUCKET}/personal_finance_ml/{particion_cartera}/valid", storage_options=STORAGE_OPTIONS)
        if particion_cartera else None
    )

    campeon, _ = registro.campeon()
    resultado = entrenar_modelo(dataset, metadata_dataset, cargar_config(), cargar_contrato(), campeon, df_cartera)
    card = resultado["card"]
    registro.guardar(card["version"], resultado["artefacto"], card, resultado["tabla_puntos"])
    if resultado["promover"]:
        registro.promover(card["version"], card["razon_promocion"])
    return {
        "version": card["version"],
        "modelo_elegido": card["modelo_elegido"],
        "auc_prueba": card["metricas_prueba"]["auc"],
        "gini_prueba": card["metricas_prueba"]["gini"],
        "promovido": resultado["promover"],
        "razon_promocion": card["razon_promocion"],
    }


def main() -> None:
    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y/%m/%d")
    print(json.dumps(entrenar_y_registrar(fecha), indent=2, default=str))


if __name__ == "__main__":
    main()
