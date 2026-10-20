"""Carga, validación y preparación de datos del componente ``lidar_box`` de Waymo v2.0.1.

Las funciones de este módulo se usan desde los notebooks y desde el resto de ``src/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
