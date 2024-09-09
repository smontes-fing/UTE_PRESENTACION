#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 10:52:31 2023
    Prepare and transform data be ready to introduce to the model
@author: seba
"""
#%%
from typing import Optional
import pandas as pd
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load
from funciones.utils import Ener_2021

import os
#%%


def transform(data, 
              train=True,
              Key_Label='Demanda', 
              columnas_feature= ['DiaSemana', 'Mes', 'Dia', 'Hora', 'Temp', 'Demanda_ener']):
    labels = pd.DataFrame()

    # Eliminar las columnas especificadas 
    if train:
        # Variable de salida
        labels = data[Key_Label]
        columnas_feature.append(Key_Label)
        
    else:
        if 'sintetic' in data.columns:
            columnas_feature.append('sintetic')    

    data_in = data.drop(columns= columnas_feature)
    
    # Cantidad de features
    features_size = len(data_in.columns)
    return data_in, labels, columnas_feature, features_size

def transform_norm(data, 
                   train=True, 
                   Key_Label='Demanda', 
                   columnas_feature=['DiaSemana', 'Mes', 'Dia', 'Hora', 'Temp'], 
                   Energia_base=0):
    # Variable de salida
    labels = pd.DataFrame()
    
    if train:
        labels[Key_Label] = data[Key_Label]
        # Escalo la labels
        if Energia_base==0:
           Energia_base = Ener_2021
        labels['Dem_norm'] = data[Key_Label] * Energia_base /data['Demanda_ener']
    else:
        labels['Dem_norm']=0
    # Eliminar las columnas especificadas 
    if 'Demanda_ener' in columnas_feature:
        columnas_feature.remove('Demanda_ener')
    data_in = data[columnas_feature]
    # Cantidad de features
    features_size = len(data_in.columns)
    return data_in, labels['Dem_norm'], columnas_feature, features_size

def transform_oneHot(data, 
                     train=True, 
                     one_hot_columns=['Mes', 'DiaMes', 'DiaSemana', 'Hora'], 
                     Key_Label='Demanda', 
                     columnas_feature=['DiaSemana', 'Mes', 'Dia', 'Hora', 'Temp']):

    labels = pd.DataFrame()
    # Encoding OneHot
    df_encoded = data
    for ohc in one_hot_columns:
        NLevels = len(data[ohc].unique())
        columna_oneHot = data[ohc]
        one_hot_encoded = tf.one_hot(columna_oneHot, depth=NLevels)
        df_encoded = pd.concat([df_encoded.reset_index(drop=True), pd.DataFrame(one_hot_encoded.numpy(), columns=[f'{ohc}_one_hot_{i}' for i in range(NLevels)])], axis=1)

    # Eliminar las columnas especificadas 
    if train:
        # Variable de salida
        labels = data[Key_Label]
        columnas_feature.append(Key_Label)
        
    columnas_feature.extend(one_hot_columns)
    data_in = df_encoded.drop(columns= columnas_feature)

    # Cantidad de features
    features_size = len(data_in.columns)
    
    return data_in, labels, columnas_feature, features_size

def transform_train(data, version, iteracion, 
                    tipo='', 
                    VAL_SPLIT= .2, 
                    seed= 42, 
                    one_hot_columns=['Mes', 'DiaMes', 'DiaSemana', 'Hora'], 
                    Key_Label='Demanda', 
                    columnas_feature=['DiaSemana', 'Mes', 'Dia', 'Hora', 'Temp', 'Demanda_ener'], 
                    Energia_base=0, 
                    scaler_path: Optional[str] = None,
                    Codigo_estacion=''): 
    train = True
    train_df, val_df = train_test_split(data, test_size= VAL_SPLIT, 
                                        shuffle=True, random_state= seed)

    if tipo == 'norm': 
        print("Transformación normalizada: Fase de entrenamiento")
        df_train, label_train, _, features_size = transform_norm(train_df, train, Key_Label, columnas_feature, Energia_base)
        # data, train=True, Key_Label='Demanda', columnas_feature=['Demanda_max', 'Demanda_mean'], ener_scal=0)
        df_val, label_val, _, features_size = transform_norm(val_df, train, Key_Label, columnas_feature, Energia_base)
    
    if tipo == 'OneHot':
        print("Transformación OneHot")
        df_train, label_train, coumnas_eliminadas, features_size = transform_oneHot (train_df, train, one_hot_columns, Key_Label, columnas_feature)
        df_val, label_val, coumnas_fature, features_size = transform_oneHot(val_df, train, one_hot_columns, Key_Label, columnas_feature)
    
    if tipo == '':
        print("Transformación normal")
        df_train, label_train, coumnas_eliminadas, features_size = transform(train_df, train, Key_Label, columnas_feature)
        df_val, label_val, coumnas_feature, features_size = transform(val_df, train, Key_Label, columnas_feature)
    
    x_train = df_train.to_numpy()
    x_val = df_val.to_numpy()
    
    # Escalar entre 0 y 1
    scaler = MinMaxScaler()                 # definimos scaler model
    scaler.fit(x_train)                     # inicializamos scaler
    data_train = scaler.transform(x_train)  # escalado training set numpy array
    data_val  = scaler.transform(x_val)     # escalado test set

    # Obtener la ruta al directorio del scaler para guardarlo
    if scaler_path is None:  
        if iteracion != '':
            # Construir la ruta completa al archivo
            if Codigo_estacion == '':
                scaler_path = 'modelos/' + f'scaler_UTE_ANN_v{version}_{iteracion}{tipo}.pkl'
            else:
                scaler_path = 'modelos/' + f'scaler_UTE_ANN_{Codigo_estacion}_v{version}_{iteracion}{tipo}.pkl'
            print(f"Guardando scaler en '{scaler_path}'...")
        else :
            scaler_path = 'modelos/'+ f'scaler_UTE_ANN_v{version}{tipo}.pkl'
            print(f"Guardando scaler en '{scaler_path}'...")
    
    try:
        # Guardar el modelo scaler en el directorio del modelo
        dump(scaler, open(scaler_path, 'wb'))
    except FileNotFoundError:
        raise FileNotFoundError('No se pudo guardar el scaler')
    
    return data_train, label_train, data_val, label_val, columnas_feature, features_size

def transform_data_input(data, 
                         version, 
                         iteracion= '', 
                         tipo='', 
                         one_hot_columns=['Mes', 'Dia', 'DiaSemana', 'Hora'], 
                         Key_Label='Demanda', 
                         columnas_feature=['DiaSemana', 'Mes', 'Dia', 'Hora','Temp', 'Demanda_ener'], 
                         Energia_base=0, 
                         scaler_path: Optional[str] = None,
                         Codigo_estacion=''
                         ): 
    train=False
    if tipo == 'norm': 
        print("Transformación normalizada: Preprocesamiento para predicción")
        data_in, _, columnas_fature, features_size = transform_norm(data, train, Key_Label, columnas_feature, Energia_base)
    elif tipo == 'OneHot':
        print("Transformación OneHot")
        data_in, _, columnas_fature, features_size = transform_oneHot(data, train, one_hot_columns, Key_Label, columnas_feature)
    elif tipo == '':
        print("Transformación normal")
        data_in, _, columnas_fature, features_size = transform(data, train, Key_Label, columnas_feature)
    else:
        raise ValueError(f'Tipo desconocido: {tipo}')

    # Obtener la ruta al directorio del script
    if scaler_path is None:
        if iteracion != '':
            # Construir la ruta completa al archivo
            if Codigo_estacion == '':
                scaler_path = 'modelos/' + f'scaler_UTE_ANN_v{version}_{iteracion}{tipo}.pkl'
            else:
                scaler_path = 'modelos/' + f'scaler_UTE_ANN_{Codigo_estacion}_v{version}_{iteracion}{tipo}.pkl'
            print(f"Buscando scaler en '{scaler_path}'...")
        else :
            scaler_path = 'modelos/'+ f'scaler_UTE_ANN_v{version}{tipo}.pkl'
            print(f"Buscando scaler en '{scaler_path}'...")
    try:
        # Cargar el modelo scaler
        scaler = load(open(scaler_path, 'rb'))
    except FileNotFoundError:
        raise FileNotFoundError(f'El archivo {scaler_path} no fue encontrado.')
    
    x_in = data_in.to_numpy()
    data_in = scaler.transform(x_in) # escalado training set numpy array
    return data_in, columnas_feature, features_size



