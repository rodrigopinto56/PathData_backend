# !pip install kagglehub
#!pip install ydata-profiling
from __future__ import annotations
from pathlib import Path
import os
import pandas as pd



# Loan Default Risk Prediction Dataset
import kagglehub
path_loan = kagglehub.dataset_download("himelsarder/loan-default-risk-prediction-dataset")
print("Path to dataset files:", path_loan)

#Personal Finance ML Dataset
import kagglehub
path_finance_ml = kagglehub.dataset_download("miadul/personal-finance-ml-dataset")
print("Path to dataset files:", path_finance_ml)

# Personal Finance Data
import kagglehub
path_finance_data = kagglehub.dataset_download("ramyapintchy/personal-finance-data")
print("Path to dataset files:", path_finance_data)

def rute_csv(file):
  files = [f for f in os.listdir(file) if f.endswith('.csv')]
  if files:
    return os.path.join(file, files[0])
  else:
    return ""

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_SOURCES ={
    "loan": os.path.join(path_loan, "loan_default_risk.csv"),
    "finance_ml": os.path.join(path_finance_ml, "personal_finance_ml.csv"),
    "finance_data": os.path.join(path_finance_data, "personal_finance_data.csv"),
}

from ydata_profiling import ProfileReport
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)


def esqueleto_diccionario(df: pd.DataFrame, nombre_dataset: str) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        serie = df[col]
        numeric = pd.api.types.is_numeric_dtype(serie)
        outliers = None
        if numeric:
            q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
            iqr = q3 - q1
            lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = int(((serie < lim_inf) | (serie > lim_sup)).sum())

        filas.append({
            "dataset": nombre_dataset,
            "columna": col,
            "tipo_dato": str(serie.dtype),
            "pct_nulos": round(serie.isnull().mean() * 100, 2),
            "valores_unicos": serie.nunique(),
            "outliers_iqr": outliers,
            "descripcion": "",
            "regla_negocio_candidata": "",
            "clasificacion": "",
        })
    return pd.DataFrame(filas)

def main() -> None:
    resumenes = []
    SOURCES = {
        "loan": rute_csv(path_loan),
        "finance_ml": rute_csv(path_finance_ml),
        "finance_data": rute_csv(path_finance_data),
    }

    for nombre, ruta in SOURCES.items():
        if not os.path.exists(ruta):
            print(f"Advertencia: No se encontró el archivo en {ruta}")
            continue

        df = pd.read_csv(ruta)
        print(f"[{nombre}] {df.shape[0]} filas, {df.shape[1]} columnas, "
              f"{int(df.duplicated().sum())} filas duplicadas")

        reporte = ProfileReport(df, title=f"Profiling - {nombre}", minimal=True)
        reporte.to_file(OUTPUT_DIR / f"{nombre}_profile.html")

        resumenes.append(esqueleto_diccionario(df, nombre))

    if resumenes:
        diccionario_completo = pd.concat(resumenes, ignore_index=True)
        diccionario_completo.to_csv(OUTPUT_DIR / "diccionario_datos_esqueleto.csv", index=False)
        print(f"\nDiccionario guardado en {OUTPUT_DIR / 'diccionario_datos_esqueleto.csv'}")

if __name__ == "__main__":
    main()

