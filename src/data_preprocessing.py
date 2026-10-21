"""Carga, validación y preparación de datos del componente ``lidar_box`` de Waymo v2.0.1.

Las funciones de este módulo se usan desde los notebooks y desde el resto de ``src/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

#: Semilla única para todo el proyecto (reproducibilidad).
RANDOM_STATE = 42

#: Códigos de la columna ``type`` según la documentación de Waymo.
NOMBRES_TIPO = {0: "Desconocido", 1: "Vehículo", 2: "Peatón", 3: "Señal", 4: "Ciclista"}

#: Columnas que deben existir después de normalizar los nombres.
COLUMNAS_REQUERIDAS = [
    "key_segment_context_name",
    "key_frame_timestamp_micros",
    "key_laser_object_id",
    "box_center_x",
    "box_center_y",
    "box_center_z",
    "box_size_x",
    "box_size_y",
    "box_size_z",
    "box_heading",
    "type",
    "num_lidar_points_in_box",
    "speed_x",
    "speed_y",
    "speed_z",
]


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los nombres de columna de Waymo a formato simple.

    Ejemplo: ``[LiDARBoxComponent].box.center.x`` pasa a ``box_center_x`` y
    ``key.laser_object_id`` pasa a ``key_laser_object_id``.

    Args:
        df: DataFrame crudo leído desde un parquet de Waymo.

    Returns:
        Una copia del DataFrame con los nombres normalizados.

    Raises:
        TypeError: si ``df`` no es un DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Se esperaba un DataFrame, se recibió {type(df).__name__}")
    df = df.copy()
    nuevos = []
    for col in df.columns:
        col = col.split("].", 1)[-1] if "]." in col else col
        nuevos.append(col.replace(".", "_"))
    df.columns = nuevos
    return df


def validar_columnas(df: pd.DataFrame, requeridas: list[str] | None = None) -> None:
    """Verifica que el DataFrame tenga todas las columnas necesarias.

    Args:
        df: DataFrame con nombres ya normalizados.
        requeridas: columnas a exigir (por defecto ``COLUMNAS_REQUERIDAS``).

    Raises:
        ValueError: si falta alguna columna.
    """
    requeridas = COLUMNAS_REQUERIDAS if requeridas is None else requeridas
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def cargar_segmento(ruta: str | Path) -> pd.DataFrame:
    """Lee un archivo parquet de ``lidar_box`` y normaliza sus columnas.

    Args:
        ruta: ruta al archivo ``.parquet`` de un segmento.

    Returns:
        DataFrame con columnas normalizadas (puede venir vacío, como en
        ``testing_location``, que no trae etiquetas).

    Raises:
        FileNotFoundError: si el archivo no existe.
        ValueError: si no es un ``.parquet`` o no se puede leer.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")
    if ruta.suffix != ".parquet":
        raise ValueError(f"Se esperaba un archivo .parquet, se recibió: {ruta.name}")
    try:
        df = pd.read_parquet(ruta)
    except Exception as exc:  # pyarrow lanza varios tipos de error
        raise ValueError(f"No se pudo leer {ruta.name}: {exc}") from exc
    return normalizar_columnas(df)


def cargar_segmentos(carpeta: str | Path, max_segmentos: int | None = None) -> pd.DataFrame:
    """Carga y concatena todos los segmentos ``.parquet`` de una carpeta.

    Los archivos sin filas se omiten con una advertencia.

    Args:
        carpeta: carpeta que contiene los parquet de ``lidar_box``.
        max_segmentos: si se indica, usa solo los primeros N archivos (orden alfabético).

    Returns:
        Un único DataFrame con las detecciones de todos los segmentos.

    Raises:
        NotADirectoryError: si ``carpeta`` no es una carpeta.
        FileNotFoundError: si no hay archivos ``.parquet``.
        ValueError: si ``max_segmentos`` no es positivo o todos los archivos están vacíos.
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {carpeta}")
    archivos = sorted(carpeta.glob("*.parquet"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .parquet en {carpeta}")
    if max_segmentos is not None:
        if max_segmentos <= 0:
            raise ValueError("max_segmentos debe ser un entero positivo")
        archivos = archivos[:max_segmentos]

    partes, vacios = [], []
    for archivo in archivos:
        df = cargar_segmento(archivo)
        if df.empty:
            vacios.append(archivo.name)
            continue
        partes.append(df)

    if vacios:
        logger.warning("Archivos sin filas omitidos: %s", vacios)
    if not partes:
        raise ValueError("Todos los archivos están vacíos; revisa que sean del split training.")

    resultado = pd.concat(partes, ignore_index=True)
    validar_columnas(resultado)
    logger.info("Cargados %d segmentos, %d filas", len(partes), len(resultado))
    return resultado


def agregar_nombre_tipo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la columna ``type_nombre`` con el nombre legible de la clase.

    Raises:
        ValueError: si el DataFrame no tiene la columna ``type``.
    """
    if "type" not in df.columns:
        raise ValueError("El DataFrame no tiene la columna 'type'")
    df = df.copy()
    df["type_nombre"] = df["type"].map(NOMBRES_TIPO)
    return df


def agregar_distancia_vehiculo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ``distancia_vehiculo``: distancia euclidiana (m) entre el vehículo y el objeto.

    Las cajas de ``lidar_box`` están expresadas en el sistema de referencia del vehículo,
    por lo que la distancia al vehículo es la norma del centro de la caja.

    Args:
        df: DataFrame con columnas ``box_center_x``, ``box_center_y`` y ``box_center_z``.

    Returns:
        Una copia del DataFrame con la columna ``distancia_vehiculo``.

    Raises:
        ValueError: si faltan las columnas del centro de la caja.
    """
    columnas = ["box_center_x", "box_center_y", "box_center_z"]
    validar_columnas(df, columnas)
    df = df.copy()
    df["distancia_vehiculo"] = np.sqrt((df[columnas] ** 2).sum(axis=1))
    return df


# ---------------------------------------------------------------------------
# Variables de entrada de cada tarea
# ---------------------------------------------------------------------------

#: Clasificación (predecir el tipo de objeto): geometría, movimiento, puntos LiDAR y distancia.
FEATURES_CLASIFICACION = [
    "box_size_x", "box_size_y", "box_size_z", "volumen",
    "rapidez", "aceleracion",
    "num_lidar_points_in_box", "num_top_lidar_points_in_box",
    "distancia_vehiculo",
]

#: Regresión (predecir la distancia): NO incluye box_center_x/y/z ni nada calculado con ellos,
#: porque con esas columnas la distancia se calcula directamente (fuga de información).
FEATURES_REGRESION = [
    "box_size_x", "box_size_y", "box_size_z", "volumen",
    "num_lidar_points_in_box", "num_top_lidar_points_in_box",
    "type",
]

#: Conteos de puntos: muy asimétricos, se les aplica log(1 + x) antes de escalar.
COLUMNAS_LOG = ["num_lidar_points_in_box", "num_top_lidar_points_in_box"]

#: Códigos de clase válidos (se descarta 0 = Desconocido).
TIPOS_VALIDOS = [1, 2, 3, 4]


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega las variables derivadas: volumen, rapidez, aceleración y distancia.

    Args:
        df: DataFrame con columnas normalizadas de ``lidar_box``.

    Returns:
        Copia del DataFrame con ``volumen``, ``rapidez``, ``aceleracion`` y
        ``distancia_vehiculo``.

    Raises:
        ValueError: si faltan columnas necesarias.
    """
    base = ["box_size_x", "box_size_y", "box_size_z",
            "speed_x", "speed_y", "speed_z",
            "acceleration_x", "acceleration_y", "acceleration_z"]
    validar_columnas(df, base)
    df = agregar_distancia_vehiculo(df)
    df["volumen"] = df["box_size_x"] * df["box_size_y"] * df["box_size_z"]
    df["rapidez"] = np.sqrt(df["speed_x"] ** 2 + df["speed_y"] ** 2 + df["speed_z"] ** 2)
    df["aceleracion"] = np.sqrt(
        df["acceleration_x"] ** 2 + df["acceleration_y"] ** 2 + df["acceleration_z"] ** 2
    )
    return df


def submuestrear_frames(df: pd.DataFrame, cada_n: int = 5) -> pd.DataFrame:
    """Conserva un frame de cada ``cada_n`` dentro de cada segmento.

    Frames consecutivos son casi idénticos (el sensor mide 10 veces por segundo), así que
    quedarse con uno de cada 5 reduce el tiempo de cómputo sin perder casi información.

    Args:
        df: DataFrame con ``key_segment_context_name`` y ``key_frame_timestamp_micros``.
        cada_n: se conserva 1 frame de cada ``cada_n`` (1 = no submuestrea).

    Returns:
        DataFrame reducido, con el índice reiniciado.

    Raises:
        ValueError: si ``cada_n`` no es un entero positivo.
    """
    if not isinstance(cada_n, int) or cada_n < 1:
        raise ValueError("cada_n debe ser un entero positivo")
    validar_columnas(df, ["key_segment_context_name", "key_frame_timestamp_micros"])
    orden = (df.groupby("key_segment_context_name")["key_frame_timestamp_micros"]
             .rank(method="dense").astype(int))
    return df[(orden - 1) % cada_n == 0].reset_index(drop=True)


def elegir_segmentos_prueba(df: pd.DataFrame, n_splits: int = 5,
                            random_state: int = RANDOM_STATE) -> list[str]:
    """Elige qué segmentos se reservan para la prueba final (aprox. 1/n_splits del total).

    Se usa ``StratifiedGroupKFold``: los segmentos van completos a un lado u otro (nunca
    se parten) y las clases quedan repartidas de forma parecida.

    Args:
        df: DataFrame con ``type`` y ``key_segment_context_name``.
        n_splits: con 5, se reserva cerca del 20 % de los segmentos.
        random_state: semilla para que la elección sea reproducible.

    Returns:
        Lista ordenada con los nombres de los segmentos de prueba.

    Raises:
        ValueError: si hay menos segmentos que ``n_splits``.
    """
    validar_columnas(df, ["type", "key_segment_context_name"])
    if df["key_segment_context_name"].nunique() < n_splits:
        raise ValueError("Hay menos segmentos que n_splits")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    _, idx_prueba = next(cv.split(df, df["type"], df["key_segment_context_name"]))
    return sorted(df.iloc[idx_prueba]["key_segment_context_name"].unique())


def dividir_por_segmentos(df: pd.DataFrame, segmentos_prueba: list[str]):
    """Separa el DataFrame en entrenamiento y prueba según los segmentos indicados.

    Returns:
        Tupla ``(df_entrenamiento, df_prueba)``.

    Raises:
        ValueError: si algún segmento de prueba no existe o si alguno de los dos queda vacío.
    """
    existentes = set(df["key_segment_context_name"].unique())
    desconocidos = set(segmentos_prueba) - existentes
    if desconocidos:
        raise ValueError(f"Segmentos de prueba que no existen en los datos: {sorted(desconocidos)}")
    es_prueba = df["key_segment_context_name"].isin(segmentos_prueba)
    if es_prueba.all() or not es_prueba.any():
        raise ValueError("La división dejó vacío el conjunto de entrenamiento o el de prueba")
    return df[~es_prueba].reset_index(drop=True), df[es_prueba].reset_index(drop=True)


def verificar_sin_fuga(df_entrenamiento: pd.DataFrame, df_prueba: pd.DataFrame) -> None:
    """Comprueba que ningún segmento ni objeto aparezca en entrenamiento y en prueba.

    Raises:
        AssertionError: si hay segmentos u objetos compartidos.
    """
    seg = set(df_entrenamiento["key_segment_context_name"]) & set(df_prueba["key_segment_context_name"])
    obj = set(df_entrenamiento["key_laser_object_id"]) & set(df_prueba["key_laser_object_id"])
    assert not seg, f"Segmentos compartidos entre entrenamiento y prueba: {sorted(seg)[:3]}..."
    assert not obj, f"Objetos compartidos entre entrenamiento y prueba: {len(obj)}"


def _filtrar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Deja solo las clases válidas (1 a 4) y reinicia el índice."""
    return df[df["type"].isin(TIPOS_VALIDOS)].reset_index(drop=True)


def preparar_clasificacion(df: pd.DataFrame):
    """Arma ``X``, ``y`` y ``grupos`` para predecir el tipo de objeto.

    Returns:
        ``X`` (variables), ``y`` (nombre de la clase) y ``grupos`` (segmento de cada fila,
        necesario para validar sin mezclar segmentos).
    """
    df = _filtrar_tipos(agregar_nombre_tipo(agregar_features(df)))
    return df[FEATURES_CLASIFICACION], df["type_nombre"], df["key_segment_context_name"]


def preparar_regresion(df: pd.DataFrame):
    """Arma ``X``, ``y`` y ``grupos`` para predecir la distancia al vehículo.

    Returns:
        ``X`` (variables, sin coordenadas), ``y`` (distancia en metros) y ``grupos``.
    """
    df = _filtrar_tipos(agregar_features(df))
    return df[FEATURES_REGRESION], df["distancia_vehiculo"], df["key_segment_context_name"]


def construir_preprocesador(tarea: str) -> ColumnTransformer:
    """Crea el transformador de variables de Scikit-learn para cada tarea.

    - Conteos de puntos LiDAR: ``log(1 + x)`` y luego escalado.
    - Demás variables numéricas: escalado estándar.
    - Regresión: el tipo de objeto se codifica con One-Hot.

    Args:
        tarea: ``"clasificacion"`` o ``"regresion"``.

    Returns:
        Un ``ColumnTransformer`` sin ajustar, listo para ir dentro de un ``Pipeline``.

    Raises:
        ValueError: si ``tarea`` no es una de las dos opciones.
    """
    if tarea == "clasificacion":
        cols_num = [c for c in FEATURES_CLASIFICACION if c not in COLUMNAS_LOG]
        cols_cat: list[str] = []
    elif tarea == "regresion":
        cols_num = [c for c in FEATURES_REGRESION if c not in COLUMNAS_LOG + ["type"]]
        cols_cat = ["type"]
    else:
        raise ValueError("tarea debe ser 'clasificacion' o 'regresion'")

    bloques = [
        ("puntos_log", Pipeline([
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("escala", StandardScaler()),
        ]), COLUMNAS_LOG),
        ("numericas", StandardScaler(), cols_num),
    ]
    if cols_cat:
        bloques.append(("tipo", OneHotEncoder(handle_unknown="ignore"), cols_cat))
    return ColumnTransformer(bloques)


def cargar_datos_modelado(carpeta: str | Path, cada_n: int = 5, n_splits: int = 5,
                          random_state: int = RANDOM_STATE):
    """Ejecuta todo el flujo de preparación y devuelve entrenamiento y prueba.

    Pasos: cargar segmentos, agregar variables, elegir segmentos de prueba (sobre todos los
    datos), submuestrear frames, dividir por segmentos y verificar que no haya fuga.

    Args:
        carpeta: carpeta con los parquet de ``lidar_box``.
        cada_n: se conserva 1 frame de cada ``cada_n``.
        n_splits: con 5, cerca del 20 % de los segmentos queda para prueba.
        random_state: semilla de la elección de segmentos.

    Returns:
        ``(df_entrenamiento, df_prueba, segmentos_prueba)``.
    """
    df = agregar_features(agregar_nombre_tipo(cargar_segmentos(carpeta)))
    segmentos_prueba = elegir_segmentos_prueba(df, n_splits, random_state)
    df_entrenamiento, df_prueba = dividir_por_segmentos(submuestrear_frames(df, cada_n), segmentos_prueba)
    verificar_sin_fuga(df_entrenamiento, df_prueba)
    return df_entrenamiento, df_prueba, segmentos_prueba
