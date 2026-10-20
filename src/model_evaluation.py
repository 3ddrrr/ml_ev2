"""Funciones de evaluación y comparación de modelos.

Se completa en el Paso 5: validación cruzada con ``GroupKFold``, métricas múltiples
(accuracy, balanced accuracy, F1 macro, R², RMSE, MAE), tiempos de entrenamiento e
inferencia, y la tabla comparativa tipo benchmark.
"""

from __future__ import annotations

from src.data_preprocessing import RANDOM_STATE  # noqa: F401
