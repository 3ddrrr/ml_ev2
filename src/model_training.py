"""Definición y entrenamiento de modelos supervisados (clasificación y regresión).

Se completa en el Paso 5 del plan: aquí vivirán los diccionarios de modelos
(Regresión Logística, KNN, Random Forest, Gradient Boosting, Ridge, etc.), los pipelines
de Scikit-learn y la función que mide el tiempo de entrenamiento de cada uno.
"""

from __future__ import annotations

from src.data_preprocessing import RANDOM_STATE  # noqa: F401  (semilla compartida)
