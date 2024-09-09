#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para entrenar una red neuronal utilizando datos de consumo energético.

Creado el 24 de abrilpyth de 2023

@author: seba
"""

import os
from typing import Optional
import pandas as pd
import tensorflow as tf
import numpy as np
from icecream import ic
from funciones.utils_transform import (
    transform_train, 
    transform_data_input
)
from funciones.utils import (
    r_squared, 
    graficar_entrenamiento, 
    validar_modelo, 
    graficar_predicciones, 
    crear_directorio,
    print_and_save_model_details
)

def load_data_CDF4(file_paths: list[str]) -> pd.DataFrame:
    """
    Carga y combina los datos desde m ltiples archivos.

    Args:
        file_paths (list[str]): Lista de rutas de archivos a cargar.

    Returns:
        pd.DataFrame: Datos combinados en un DataFrame de pandas.
    """
    start_time = tm.time()
    dfs: list[pd.DataFrame] = []
    for file_path in file_paths:
        try:
            ds = xr.open_dataset(file_path, chunks='auto')
            df = ds.to_dataframe().reset_index()
            dfs.append(df)
            del ds
        except Exception as e:
            ds = None
            raise FileNotFoundError(f"Error al cargar el archivo {file_path}: {e}")
    end_time = tm.time()
    # Medir el tiempo de carga
    load_time = end_time - start_time
    print(f"Tiempo de carga del archivo: {load_time:.2f} segundos")

    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined_df['t2m'] = combined_df['t2m'] - 273.15  # Convertir de Kelvin a Celsius    


    return combined_df

def cargar_datos(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los datos de entrenamiento desde un archivo CSV y los separa en datos de entrenamiento y prueba.

    Parameters:
        data_path (str): Ruta al archivo CSV con los datos.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Dos DataFrames, el primero es el de entrenamiento 
        y el segundo es el de prueba (a o 2021).
    """
    df = pd.read_csv(data_path, skipinitialspace=True)
    data_test = df[df['Ano'] == 2021].reset_index(drop=True)
    E_base = data_test['Demanda_ener'].iloc[1] # Valor de la demanda en 2021
    data_train = df[df['Ano'] != 2021].reset_index(drop=True)
    return data_train, data_test, E_base

def preparar_datos(
    data_train: pd.DataFrame, 
    version: int, 
    epochs: int, 
    tipo: str, 
    val_split: float, 
    Energia_base: float,
    seed: int,
    scaler_path: Optional[str] = None,
) -> tuple[
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        np.ndarray, 
        int
        ]:
    """
    Transforma los datos para que puedan ser utilizados como insumo en el modelo.

    Parameters:
        data_train (pd.DataFrame): Datos de entrenamiento.
        version (int): Versión de la transformación.
        epochs (int): Número de iteraciones (epochs).
        tipo (str): Tipo de transformación (ej. 'norm', 'OneHot').
        val_split (float): Porcentaje de datos a usar para validación.
        Energia_base (float): Energ[ia base con la que entrena el modelo, Energia consumida en el año 2021.]
        seed (int): Semilla para la división de los datos.
        scaler_path (str): Ruta del scaler para normalización.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: 
            - data_train (np.ndarray): Datos de entrenamiento transformados.
            - label_train (np.ndarray): Etiquetas de entrenamiento transformadas.
            - data_val (np.ndarray): Datos de validación transformados.
            - label_val (np.ndarray): Etiquetas de validación transformadas.
            - features_size (int): Tamaño de entrada del modelo.
    """
    (data_train, 
    label_train, 
    data_val, 
    label_val, 
    _, 
    features_size) = transform_train(
                        data_train, 
                        version, 
                        epochs, 
                        tipo, 
                        val_split, 
                        seed,
                        Energia_base= Energia_base,
                        scaler_path=scaler_path,
                        )
    ic("VALORES NORMALIZADOS DE LOS DATOS DE ENTRENAMIENTO")
    ic(data_train[1:5, :])
    return data_train, label_train, data_val, label_val, features_size

def crear_modelo(
    input_shape: int,
    dense_layer: int,
    layer_size: int,
    dropout: float,
    learning_rate: float,
    decay: float
) -> tf.keras.Sequential:
    """
    Crea y compila el modelo de red neuronal.

    Parameters:
        input_shape (int): Tamaño de la capa de entrada.
        dense_layer (int): Número de capas densas.
        layer_size (int): Número de neuronas por capa.
        dropout (float): Tasa de dropout.
        learning_rate (float): Tasa de aprendizaje.
        decay (float): Decaimiento del paso de aprendizaje.

    Returns:
        model (tf.keras.Sequential): Modelo compilado.
    """
    model = tf.keras.Sequential(name='Modelo_ANN')
    
    # Capa de entrada
    model.add(tf.keras.layers.Dense(layer_size, activation="relu", input_shape=(input_shape,), bias_initializer='ones'))
    model.add(tf.keras.layers.Dropout(rate=dropout))

    # Capas ocultas
    for _ in range(dense_layer - 1):
        model.add(tf.keras.layers.Dense(layer_size, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=dropout))

    # Capa de salida
    model.add(tf.keras.layers.Dense(1, kernel_initializer='normal', activation='relu'))

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compilación del modelo
    model.compile(loss="mse", optimizer=opt, metrics=[r_squared, 'mape'])
    return model

def entrenar_modelo(
    model: tf.keras.Sequential,
    data_train: np.ndarray,
    label_train: np.ndarray,
    data_val: np.ndarray,
    label_val: np.ndarray,
    epochs: int,
    batch_size: int
) -> tf.keras.callbacks.History:
    """
    Entrena el modelo utilizando los datos de entrenamiento y validación.

    Parameters:
        model (tf.keras.Sequential): Modelo a entrenar.
        data_train (ndarray): Datos de entrada para el entrenamiento.
        label_train (ndarray): Etiquetas para el entrenamiento.
        data_val (ndarray): Datos de entrada para la validación.
        label_val (ndarray): Etiquetas para la validación.
        epochs (int): Número de épocas de entrenamiento.
        batch_size (int): Tamaño del batch.

    Returns:
        history (tf.keras.callbacks.History): Historia del entrenamiento.
    """
    history = model.fit(
        data_train, 
        label_train, 
        batch_size=batch_size, 
        epochs=epochs, 
        validation_data=(data_val, label_val), 
        verbose=2
    )
    return history

def guardar_modelo(model: tf.keras.Sequential, model_path: str) -> None:
    """
    Guarda el modelo entrenado en un archivo.

    Parameters:
        model (tf.keras.Sequential): Modelo entrenado.
        model_path (str): Ruta donde guardar el modelo.
    Returns:
        None
    """
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
        ic(f"Directorio {os.path.dirname(model_path)} creado con exito. Guardando el modelo ...")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")

def main():
    # Definir parámetros
    data_path = 'datos/datosANN2008-2021.csv'
    model_path = 'modelos/UTE_ANN_test'
    scaler_path = 'modelos/UTE_ANN_test/scaler_UTE_ANN_test.pkl'
    output_dir = model_path + '/entrenamiento'
    version = 3
    epochs = 200
    val_split = 0.2
    seed = 42
    dense_layer = 5
    layer_size = 1024
    dropout = 0.1
    learning_rate = 0.001
    decay = 1e-6
    batch_size = 256

    # Cargar y preparar los datos, Energia_base es el valor de la demanda en 2021
    data_train, data_test, Energia_base = cargar_datos(data_path)
    print(data_test.head(5))
    
    (
    data_train, 
    label_train, 
    data_val, 
    label_val, 
    features_size
        ) = preparar_datos(
                            data_train, 
                            version, 
                            epochs, 
                            'norm', 
                            val_split, 
                            Energia_base,
                            seed,
                            escaler_path=scaler_path,
                            )
    # Crear y entrenar el modelo
    model = crear_modelo(
                        features_size, 
                        dense_layer, 
                        layer_size, 
                        dropout, 
                        learning_rate, 
                        decay)
    print(label_train.head(5))
    print(label_val.head(5))
    history = entrenar_modelo(
                                model, 
                                data_train, 
                                label_train, 
                                data_val, 
                                label_val,
                                epochs, 
                                batch_size
                            )

    # Guardar el modelo entrenado
    guardar_modelo(model, model_path)

    # Graficar resultados del entrenamiento
    graficar_entrenamiento(history, output_dir)

    # Valoido el modelo contra los datos de 2021
    print("DATOS DE TEST, valores reales 2021")
    print(data_test.head(5))
    label_test = pd.DataFrame()
    label_test["Dem_norm"] = data_test['Demanda'] * Energia_base / data_test['Demanda_ener']
    data_test, _, _ = transform_data_input(
                                    data_test, 
                                    version, 
                                    epochs, 
                                    'norm', 
                                    Energia_base=Energia_base,
                                    )
    pred_norm = validar_modelo(
                                model, 
                                data_test, 
                                label_test, 
                                output_dir, 
                                'norm'
                               )
    
    graficar_predicciones(pred_norm, label_test, output_dir)

    # Guardar detalles del modelo
    print_and_save_model_details(model, output_dir=output_dir)



    


if __name__ == "__main__":
    main()
