"""Evaluación y comparación de modelos con validación cruzada.

Para cada modelo y cada fold se mide: varias métricas de calidad y dos tiempos
(entrenamiento y predicción, medidos por separado). Luego se resume todo como media y
desviación entre folds.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from src.model_training import construir_pipeline

#: Clases y el nombre corto que se usa en las columnas de resultados.
CLASES = {"Vehículo": "vehiculo", "Señal": "senal", "Peatón": "peaton", "Ciclista": "ciclista"}


def metricas_clasificacion(y_real, y_pred, clases_proba, proba) -> dict[str, float]:
    """Calcula las métricas de clasificación de un fold.

    Args:
        y_real: clases reales.
        y_pred: clases predichas.
        clases_proba: orden de las clases en las columnas de ``proba`` (``modelo.classes_``).
        proba: probabilidades predichas, una columna por clase.

    Returns:
        Diccionario con accuracy, balanced accuracy, F1 macro, precision macro, ROC AUC (OvR)
        y el recall de cada clase. Si en el fold falta alguna clase, el ROC AUC queda en NaN.
    """
    m = {
        "accuracy": accuracy_score(y_real, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "f1_macro": f1_score(y_real, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_real, y_pred, average="macro", zero_division=0),
    }
    try:
        m["roc_auc_ovr"] = roc_auc_score(y_real, proba, multi_class="ovr", labels=list(clases_proba))
    except ValueError:  # falta alguna clase en el fold: el AUC no se puede calcular
        m["roc_auc_ovr"] = np.nan
    for clase, corto in CLASES.items():
        m[f"recall_{corto}"] = recall_score(y_real, y_pred, labels=[clase], average="macro",
                                            zero_division=0)
    return m


def metricas_regresion(y_real, y_pred) -> dict[str, float]:
    """Calcula R², RMSE y MAE de un fold (los errores quedan en metros)."""
    return {
        "r2": r2_score(y_real, y_pred),
        "rmse": root_mean_squared_error(y_real, y_pred),
        "mae": mean_absolute_error(y_real, y_pred),
    }


def evaluar_modelos(modelos: dict, tarea: str, X: pd.DataFrame, y: pd.Series,
                    grupos: pd.Series, cv) -> pd.DataFrame:
    """Evalúa cada modelo con validación cruzada agrupada.

    Los folds se calculan una sola vez, así todos los modelos se evalúan exactamente con
    los mismos datos (comparación justa). El ciclo se hace a mano, con el mismo esquema que
    ``cross_validate``, para medir con exactitud cuánto demora entrenar y cuánto demora predecir.
    Los modelos de cada fold no se guardan, así se usa poca memoria.

    Args:
        modelos: diccionario ``{nombre: modelo sin entrenar}``.
        tarea: ``"clasificacion"`` o ``"regresion"``.
        X, y: variables de entrada y objetivo.
        grupos: segmento de cada fila (los segmentos no se parten entre folds).
        cv: objeto de validación cruzada (por ejemplo ``StratifiedGroupKFold``).

    Returns:
        Un DataFrame con una fila por modelo y fold: métricas, tiempo de entrenamiento (s)
        y tiempo de predicción (ms por cada 1.000 filas).

    Raises:
        ValueError: si no hay modelos, si la tarea no es válida o si ``X``, ``y`` y ``grupos``
            no tienen el mismo largo.
    """
    if tarea not in ("clasificacion", "regresion"):
        raise ValueError("tarea debe ser 'clasificacion' o 'regresion'")
    if not modelos:
        raise ValueError("El diccionario de modelos está vacío")
    if not (len(X) == len(y) == len(grupos)):
        raise ValueError("X, y y grupos deben tener el mismo número de filas")

    folds = list(cv.split(X, y, grupos))
    filas = []
    inicio_total = time.perf_counter()

    for nombre, modelo in modelos.items():
        inicio = time.perf_counter()
        for k, (idx_ent, idx_val) in enumerate(folds, start=1):
            X_ent, y_ent = X.iloc[idx_ent], y.iloc[idx_ent]
            X_val, y_val = X.iloc[idx_val], y.iloc[idx_val]

            pipe = construir_pipeline(tarea, modelo)
            t0 = time.perf_counter()
            pipe.fit(X_ent, y_ent)
            t_entrenamiento = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = pipe.predict(X_val)
            t_prediccion = time.perf_counter() - t0

            if tarea == "clasificacion":
                metricas = metricas_clasificacion(y_val, y_pred, pipe.classes_, pipe.predict_proba(X_val))
            else:
                metricas = metricas_regresion(y_val, y_pred)

            filas.append({"modelo": nombre, "fold": k,
                          "tiempo_entrenamiento_s": t_entrenamiento,
                          "tiempo_prediccion_ms_por_1000": t_prediccion * 1e6 / len(idx_val),
                          **metricas})
        print(f"  {nombre}: {time.perf_counter() - inicio:.1f} s")

    print(f"Tiempo total: {time.perf_counter() - inicio_total:.1f} s")
    return pd.DataFrame(filas)


def resumir_resultados(detalle: pd.DataFrame) -> pd.DataFrame:
    """Resume el detalle por fold: media y desviación estándar de cada métrica por modelo.

    Returns:
        DataFrame indexado por modelo (en el orden original) con columnas
        ``<metrica>_media`` y ``<metrica>_desv``.
    """
    columnas = [c for c in detalle.columns if c not in ("modelo", "fold")]
    orden = detalle["modelo"].drop_duplicates()
    grupo = detalle.groupby("modelo")[columnas]
    resumen = pd.concat([grupo.mean().add_suffix("_media"), grupo.std().add_suffix("_desv")], axis=1)
    resumen = resumen.reindex(orden)
    return resumen[[f"{c}_{s}" for c in columnas for s in ("media", "desv")]]


def tabla_legible(resumen: pd.DataFrame, metricas: list[str], decimales: int = 3) -> pd.DataFrame:
    """Devuelve una tabla de texto con el formato ``media ± desviación`` para mostrar."""
    tabla = pd.DataFrame(index=resumen.index)
    for m in metricas:
        tabla[m] = (resumen[f"{m}_media"].round(decimales).astype(str)
                    + " ± " + resumen[f"{m}_desv"].round(decimales).astype(str))
    return tabla


def guardar_resultados(tabla: pd.DataFrame, ruta: str | Path) -> Path:
    """Guarda una tabla de resultados en CSV, creando la carpeta si no existe."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(ruta, encoding="utf-8")
    return ruta
