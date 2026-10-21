"""Definición de los modelos supervisados y construcción de sus pipelines.

Cada modelo se entrega con su configuración inicial (la de Scikit-learn, salvo lo indicado).
En el notebook 04 estas configuraciones se optimizan con GridSearchCV y RandomizedSearchCV.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data_preprocessing import RANDOM_STATE, construir_preprocesador

NOMBRE_BASE_CLASIFICACION = "Línea base (clase más frecuente)"
NOMBRE_BASE_REGRESION = "Línea base (distancia promedio)"


def modelos_clasificacion(random_state: int = RANDOM_STATE,
                          incluir_base: bool = True) -> dict[str, BaseEstimator]:
    """Devuelve los modelos de clasificación con su configuración inicial.

    Args:
        random_state: semilla de los modelos que usan azar.
        incluir_base: si es True, agrega la línea base (siempre la clase más frecuente).

    Returns:
        Diccionario ``{nombre: modelo}`` (modelos sin entrenar).
    """
    modelos: dict[str, BaseEstimator] = {}
    if incluir_base:
        modelos[NOMBRE_BASE_CLASIFICACION] = DummyClassifier(strategy="most_frequent")
    modelos.update({
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Árbol de Decisión": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                random_state=random_state),
        "Gradient Boosting": HistGradientBoostingClassifier(early_stopping=False,
                                                            random_state=random_state),
    })
    return modelos


def modelos_regresion(random_state: int = RANDOM_STATE,
                      incluir_base: bool = True) -> dict[str, BaseEstimator]:
    """Devuelve los modelos de regresión con su configuración inicial.

    Args:
        random_state: semilla de los modelos que usan azar.
        incluir_base: si es True, agrega la línea base (siempre la distancia promedio).

    Returns:
        Diccionario ``{nombre: modelo}`` (modelos sin entrenar).
    """
    modelos: dict[str, BaseEstimator] = {}
    if incluir_base:
        modelos[NOMBRE_BASE_REGRESION] = DummyRegressor(strategy="mean")
    modelos.update({
        "Regresión Lineal (Ridge)": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "Árbol de Decisión": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                               random_state=random_state),
        "Gradient Boosting": HistGradientBoostingRegressor(early_stopping=False,
                                                           random_state=random_state),
    })
    return modelos


def construir_pipeline(tarea: str, modelo: BaseEstimator) -> Pipeline:
    """Une el preprocesador de la tarea con un modelo, en un solo Pipeline.

    Se clona el modelo, así el original nunca queda entrenado por accidente.

    Args:
        tarea: ``"clasificacion"`` o ``"regresion"``.
        modelo: modelo de Scikit-learn sin entrenar.

    Returns:
        ``Pipeline`` con los pasos ``prep`` y ``modelo``.

    Raises:
        ValueError: si la tarea no es válida.
    """
    if tarea not in ("clasificacion", "regresion"):
        raise ValueError("tarea debe ser 'clasificacion' o 'regresion'")
    return Pipeline([("prep", construir_preprocesador(tarea)), ("modelo", clone(modelo))])
