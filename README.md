# EP2 — Modelos supervisados y no supervisados sobre detecciones LiDAR (Waymo v2.0.1)

> Documento en construcción: se completa a medida que avanza cada paso del plan.

## 1. Objetivo
TODO: comparar modelos de clasificación (tipo de objeto), regresión (distancia entre el
vehículo y cada objeto) y aprendizaje no supervisado, evaluando calidad y velocidad.

## 2. Datos
- Dataset: Waymo Open Dataset v2.0.1 (Perception, Modular, sin mapas), componente `lidar_box`.
- Segmentos usados: TODO (n.º y split `training`).
- Los `.parquet` **no se suben al repositorio** (licencia de Waymo). Ver `data/raw/`.

## 3. Estructura del proyecto
```
notebooks/   01 a 05: exploración, modelado, evaluación, optimización, análisis final
src/         data_preprocessing.py, model_training.py, model_evaluation.py, hyperparameter_tuning.py
data/raw/    parquet descargados de Waymo (no versionados)
models/trained_models/   modelos serializados (joblib)
results/     metrics/, plots/, reports/
```

## 4. Cómo reproducir
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt
# colocar los parquet de lidar_box en data/raw/lidar_box/
# ejecutar los notebooks en orden: 01 -> 05
```
Todos los procesos aleatorios usan `RANDOM_STATE = 42` (definido en `src/data_preprocessing.py`).

## 5. Resultados
TODO.
