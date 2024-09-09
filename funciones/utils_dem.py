#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 09:59:06 2023

@author: seba
"""

#%% Importo librerias

import pandas as pd
import datetime
from itertools import product
import warnings 
# from ent_modelo import entreno_Modelo, r_squared
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from typing import List
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import MeanMetricWrapper
from pickle import dump, load
import sqlite3

import sys
sys.path.append("../funciones")
from funciones.utils_temp import obtener_temperaturas_cercanas
from funciones.utils import r_squared, Ener_2021
from funciones.utils_transform import transform_data_input

#%% Cargar y procesar datos de demanda
def cargar_procesar_datos_demanda(dir_demanda, anio_ini, anio_fin):
    # Cargar datos de demanda
    demanda = pd.read_csv(dir_demanda)
    demanda['Fecha'] = pd.to_datetime(demanda['Fecha'])

    # Generar variables auxiliares
    demanda['Ano']  = demanda['Fecha'].dt.year
    demanda['DiaSemana'] = demanda['Fecha'].dt.weekday.astype(str)

    # Tomar un subconjunto de años
    demanda = demanda.drop(demanda[demanda['Ano'] < anio_ini].index)
    demanda = demanda.drop(demanda[demanda['Ano'] > anio_fin].index)
    
    if demanda.isnull().values.any(): 
        warnings.warn("El set de datos de demanda generado contiene {} valores 'NA'.\nSe eliminarán estos registros del set de datos.".format(demanda.isna().sum()))
    
    # Verifico que no haya valores NA (not available)
    demanda = demanda.dropna()
    
    # Generar variables auxiliares a partir de lso datos de demanda
    demandas_maximas  = demanda.groupby('Ano')['Demanda'].max()
    demandas_minimas  = demanda.groupby('Ano')['Demanda'].min()
    demandas_media    = demanda.groupby('Ano')['Demanda'].mean()
    demandas_energia  = demanda.groupby('Ano')['Demanda'].sum()

    # Agrego las columnas de demanda máxima y mínima al DataFrame original
    demanda = demanda.merge(demandas_maximas, on='Ano', suffixes=('', '_max'))
    demanda = demanda.merge(demandas_minimas, on='Ano', suffixes=('', '_min'))
    demanda = demanda.merge(demandas_energia, on='Ano', suffixes=('', '_ener'))
    demanda = demanda.merge(demandas_media  , on='Ano', suffixes=('', '_mean'))
    
    return(demanda)

#%% Cargar y procesar datos de demanda desde BD

def cargar_procesar_datos_demanda_2(dir_demanda, codigo_estacion, anio_ini, anio_fin):
    
    con = sqlite3.connect(dir_demanda)
    cur = con.cursor()

    # Cargar datos de demanda
    demanda =  pd.read_sql_query("SELECT FechaHum AS Fecha, Codigo_estacion, Cambio_horario, Valor as Demanda FROM ESTACION_HORA_HUMANA WHERE Codigo_estacion = '{}';".format(codigo_estacion), con)
    con.close()
    
    demanda['Fecha'] = pd.to_datetime(demanda['Fecha'])

    # Generar variables auxiliares
    demanda['Ano']  = demanda['Fecha'].dt.year
    demanda['DiaSemana'] = demanda['Fecha'].dt.weekday.astype(str)

    # Tomar un subconjunto de años
    demanda = demanda.drop(demanda[demanda['Ano'] < anio_ini].index)
    demanda = demanda.drop(demanda[demanda['Ano'] > anio_fin].index)
    
    if demanda.isnull().values.any(): 
        warnings.warn("El set de datos de demanda generado contiene {} valores 'NA'.\nSe eliminarán estos registros del set de datos.".format(demanda.isna().sum()))
    
    # Verifico que no haya valores NA (not available)
    demanda = demanda.dropna()
    
    # Generar variables auxiliares a partir de lso datos de demanda
    demandas_maximas  = demanda.groupby('Ano')['Demanda'].max()
    demandas_minimas  = demanda.groupby('Ano')['Demanda'].min()
    demandas_media    = demanda.groupby('Ano')['Demanda'].mean()
    demandas_energia  = demanda.groupby('Ano')['Demanda'].sum()

    # Agrego las columnas de demanda máxima y mínima al DataFrame original
    demanda = demanda.merge(demandas_maximas, on='Ano', suffixes=('', '_max'))
    demanda = demanda.merge(demandas_minimas, on='Ano', suffixes=('', '_min'))
    demanda = demanda.merge(demandas_energia, on='Ano', suffixes=('', '_ener'))
    demanda = demanda.merge(demandas_media  , on='Ano', suffixes=('', '_mean'))
    
    return(demanda)

    
#%% Cargar y procesar datos de temperatura

def cargar_procesar_datos_temp(dir_temp, anio_ini, anio_fin):
    # Cargar datos de temperatura
    temp = pd.read_csv(dir_temp)

    # Generar variables a partir de datos del calendario
    temp['Ano'] = temp['Ano'].astype(int)
    temp['Mes'] = temp['Mes'].astype(int)
    temp['DiaMes'] = temp['DiaMes'].astype(int)
    temp['Hora'] = temp['Hora'].astype(int)
    temp['Fecha'] = pd.to_datetime(dict(year=temp.Ano, month=temp.Mes, day=temp.DiaMes, hour=temp.Hora))
    temp['Fecha_dia'] = pd.to_datetime(dict(year=temp.Ano, month=temp.Mes, day=temp.DiaMes))
    
    # Me quedo solo con la temperatura registrada en el punto Long=-56.25 y lat=34.75 (Carrasco)
    temp            = temp.drop(temp[temp['longitud'] != -56.25].index)
    temp_filtrada   = temp[(temp['Ano'] >= anio_ini) & (temp['Ano'] <= anio_fin)]
    temp_filtrada   = temp_filtrada.reset_index(drop = True)

    # Hallar temperatura máxima, mínima y media de cada día
    temps_maximas  = temp_filtrada.groupby('Fecha_dia')['Temp'].max()
    temps_minimas  = temp_filtrada.groupby('Fecha_dia')['Temp'].min()
    temps_medias   = temp_filtrada.groupby('Fecha_dia')['Temp'].mean()

    # Agrego las columnas de temperaturas máxima, mínima y media de cada día
    temp_filtrada = temp_filtrada.merge(temps_maximas, on='Fecha_dia', suffixes=('', '_max'))
    temp_filtrada = temp_filtrada.merge(temps_minimas, on='Fecha_dia', suffixes=('', '_min'))
    temp_filtrada = temp_filtrada.merge(temps_medias, on='Fecha_dia', suffixes=('', '_mean'))

    # Potencias de las temperaturas
    temp_filtrada['Temp_2'] = temp_filtrada['Temp'] ** 2
    temp_filtrada['Temp_3'] = temp_filtrada['Temp'] ** 3

    return(temp_filtrada)

#%% Cargar y procesar datos de temperatura

def cargar_procesar_datos_temp_2(anio_ini, anio_fin, latitud, longitud):
    
    # Cargar datos de temperatura
    temp = obtener_temperaturas_cercanas(latitud, longitud)

    # Generar variables a partir de datos del calendario
    temp['Ano'] = temp['Ano'].astype(int)
    temp['Mes'] = temp['Mes'].astype(int)
    temp['DiaMes'] = temp['DiaMes'].astype(int)
    temp['Hora'] = temp['Hora'].astype(int)
    temp['Fecha'] = pd.to_datetime(dict(year=temp.Ano, month=temp.Mes, day=temp.DiaMes, hour=temp.Hora))
    temp['Fecha_dia'] = pd.to_datetime(dict(year=temp.Ano, month=temp.Mes, day=temp.DiaMes))

    temp = temp.rename(columns={"t2m":"Temp"})

    temp_filtrada   = temp[(temp['Ano'] >= anio_ini) & (temp['Ano'] <= anio_fin)]
    temp_filtrada   = temp_filtrada.reset_index(drop = True)

    # Hallar temperatura máxima, mínima y media de cada día
    temps_maximas  = temp_filtrada.groupby('Fecha_dia')['Temp'].max()
    temps_minimas  = temp_filtrada.groupby('Fecha_dia')['Temp'].min()
    temps_medias   = temp_filtrada.groupby('Fecha_dia')['Temp'].mean()

    # Agrego las columnas de temperaturas máxima, mínima y media de cada día
    temp_filtrada = temp_filtrada.merge(temps_maximas, on='Fecha_dia', suffixes=('', '_max'))
    temp_filtrada = temp_filtrada.merge(temps_minimas, on='Fecha_dia', suffixes=('', '_min'))
    temp_filtrada = temp_filtrada.merge(temps_medias, on='Fecha_dia', suffixes=('', '_mean'))

    # Potencias de las temperaturas
    temp_filtrada['Temp_2'] = temp_filtrada['Temp'] ** 2
    temp_filtrada['Temp_3'] = temp_filtrada['Temp'] ** 3
    
    return(temp_filtrada)

#%% Cargar y procesar datos de PBI

def cargar_procesar_datos_pbi(dir_pbi):
    # Cargar datos de PBI
    pbi = pd.read_csv(dir_pbi, sep=';')
    pbi.rename(columns = {'valor':'PBI'}, inplace = True)
    pbi.rename(columns = {'fecha':'Fecha'}, inplace = True)
    pbi['Fecha'] = pd.to_datetime(pbi['Fecha'])

    return(pbi)
 
#%% Cargar y procesar datos de feriados

def cargar_procesar_datos_feriados(dir_feriados):
    # Cargar datos de feriados
    feriados = pd.read_csv(dir_feriados)
    feriados['Fecha'] = pd.to_datetime(feriados['Fecha'])

    return(feriados)
 
#%% Genero un dataframe a partir de las rutas a los datos de demanda, temperatura, PBI y feriados. Tomo un subset entre los años anio_ini y anio_fin.

def cargar_set_datos(dir_demanda, dir_temp, dir_pbi, dir_feriados, anio_ini, anio_fin):
    # Cargar datos procesados de demanda
    demanda = cargar_procesar_datos_demanda(dir_demanda, anio_ini, anio_fin)
    
    # Cargar datos procesados de temperatura
    temp = cargar_procesar_datos_temp(dir_temp, anio_ini, anio_fin)
    
    # Cargar datos de pbi procesados
    pbi = cargar_procesar_datos_pbi(dir_pbi)
    
    # Cargar datos de feriados procesados
    feriados = cargar_procesar_datos_feriados(dir_feriados)
    
    # Combinar datos y generar el set
    data = demanda.drop(columns=['Ano', 'Codigo_estacion'])
    data = data.merge(right=temp.drop(columns=['longitud', 'latitud']), on='Fecha')
    data = data.merge(right=pbi, on='Fecha')
    data['Feriado'] = 0
    data.loc[data['Fecha_dia'].isin(feriados['Fecha']), 'Feriado'] = 1

    return(data.drop(columns=['Fecha', 'Fecha_dia']))

def cargar_set_datos_2(dir_demanda, dir_pbi, dir_feriados, anio_ini, anio_fin, codigo_estacion):
    # Cargar datos procesados de demanda
    demanda = cargar_procesar_datos_demanda_2(dir_demanda, codigo_estacion, anio_ini, anio_fin)

    # Coordendas de la estacion
    con = sqlite3.connect(dir_demanda)
    cur = con.cursor()

    # Cargar datos de demanda
    if codigo_estacion == 'SUM':
        coordenadas = pd.read_sql_query("SELECT * FROM ESTACIONES WHERE Codigo_estacion = 'MVE';", con)
    else:
        coordenadas = pd.read_sql_query("SELECT * FROM ESTACIONES WHERE Codigo_estacion = '{}';".format(codigo_estacion), con)
    con.close()

    # Cargar datos procesados de temperatura
    temp = cargar_procesar_datos_temp_2(anio_ini, anio_fin, coordenadas.Coord_latitud, coordenadas.Coord_longitud)

    # Cargar datos de pbi procesados
    pbi = cargar_procesar_datos_pbi(dir_pbi)

    # Cargar datos de feriados procesados
    feriados = cargar_procesar_datos_feriados(dir_feriados)

    # Combinar datos y generar el set
    data = demanda.drop(columns=['Ano', 'Codigo_estacion'])
    data = data.merge(right=temp, on='Fecha')
    data = data.merge(right=pbi, on='Fecha')
    data['Feriado'] = 0
    data.loc[data['Fecha_dia'].isin(feriados['Fecha']), 'Feriado'] = 1

    return(data.drop(columns=['Fecha', 'Fecha_dia', 'time']))

#%% Cargar csv de datos simulados

def cargar_datos_dem_sint(
        data_path: str,
        ener_proy: float,
        ener_base: float
        ) -> pd.DataFrame:
    """
    Función para cargar datos de demanda sintética desde una ruta de datos especificada.
    
    Parámetros:
        data_path (str): La ruta al archivo CSV que contiene los datos de demanda sintética.
        ener_proy (float): La energía proyectada para el año que se va a simular.
        ener_base (float): La energía base para el año que se va a simular.
    
    Retorno:
        pandas.DataFrame: Los datos de demanda sintética cargados como un DataFrame de pandas.
    """
    try:
        data = pd.read_csv(data_path, engine='pyarrow')
        if data_path == 'datos/dem_sintetico_100.csv': 
            data = data * (ener_proy/ener_base)
    except FileNotFoundError:
        raise FileNotFoundError(f'Error: la ruta {data_path} para'+ 
                                'levantar los datos de demanda sintetica no fue encontrado.')
    return data

# Genero un dataframe a partir de datos de tempratura simulados

def genero_ano_dem_sint(data_temp: pd.DataFrame, 
                        ano_to_sim: int, 
                        model: tf.keras.models, 
                        scaler_path: str,
                        tipo_modelo: str, 
                        version: int, 
                        iteracion: int, 
                        ener: float) -> pd.DataFrame:
    '''
    La función recibe un data frame de N años sinteticos de temperatura y realiza la 
    predicción de la demanda, asumiendo que la energía para ese año es _ener_

    Parameters
    ----------
    data_temp : pd.DataFrame
        DESCRIPTION.
    ano_to_sim : int
        DESCRIPTION.
    model : keras.Model
        DESCRIPTION.
    tipo_modelo : str
        DESCRIPTION.
    version : int
        DESCRIPTION.
    iteracion : int
        DESCRIPTION.
    ener : float
        DESCRIPTION.

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.
    '''
    if ener==0:
        data_temp['Demanda_ener'] = Ener_2021  # Valor de energía año 2021
    else: 
        data_temp['Demanda_ener'] = ener
    df_pred = pd.DataFrame()
    for ano in range(ano_to_sim):
        col = 'Ano_' + str(ano)
        data_raw = pd.DataFrame()
        columnas= ['DiaSemana', 'Mes', 'Dia', 'Hora', col, 'Demanda_ener']
        data_raw = data_temp[columnas]
        data_in, _, _, _ = transform_data_input(data_raw, 
                                       version, 
                                       iteracion, 
                                       tipo=tipo_modelo, 
                                       one_hot_columns=['Mes', 'Dia', 'DiaSemana', 'Hora'], 
                                       Key_Label='Demanda', 
                                       columnas_feature=columnas, 
                                       ener_scal=0, 
                                       scaler_path=scaler_path,
                                       Codigo_estacion='',
                                       )
        df_pred[col]=pd.DataFrame(model.predict(data_in,
                              # batch_size=None,
                              # verbose='auto',
                              # steps=None,
                              # callbacks=None,
                              # max_queue_size=10,
                              # workers=1,
                              # use_multiprocessing=False
                                                        )
                                  
                                        )
        # Escalado
        if tipo_modelo != '':
            df_pred[col] = df_pred[col] * (ener/Ener_2021)
    return df_pred




