#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:18:49 2023

@author: seba
"""


#%%
import pandas as pd
import datetime
# from itertools import product
# from ent_modelo import entreno_Modelo, r_squared
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.metrics import MeanMetricWrapper
# from pickle import dump, load
import random
import netCDF4
import xarray as xr
# import dask
import math


#%% Funciones Auxiliares 
#Función para separar la fecha en año, mes, día y hora y además agregar el día acumulado del año y el número de semana correspodiente

def adecuar_dataframe_2(A):

    # Convertir la columna fecha en un objeto datetime
    A['time'] = pd.to_datetime(A['time'])

    # Crear nuevas columnas para año, mes, día y hora
    A['Ano'] = A['time'].apply(lambda x: x.year)
    A['Mes'] = A['time'].apply(lambda x: x.month)
    A['DiaMes'] = A['time'].apply(lambda x: x.day)
    A['Hora'] = A['time'].apply(lambda x: x.hour)

    A = A[['time', 'Ano', 'Mes','DiaMes','Hora', 't2m']]
    A = A.sort_values('time')

    return A

def adecuar_dataframe(A):

    # Convertir la columna fecha en un objeto datetime
    A['time'] = pd.to_datetime(A['time'])

    # Crear nuevas columnas para año, mes, día y hora
    A['Ano'] = A['time'].apply(lambda x: x.year)
    A['Mes'] = A['time'].apply(lambda x: x.month)
    A['Dia'] = A['time'].apply(lambda x: x.day)
    A['Hora'] = A['time'].apply(lambda x: x.hour)

    A = A[['time', 'Ano', 'Mes','Dia','Hora', 't2m']]
    
    year_day = []
    for i in range(len(A)):
        day = A['time'].iloc[i].timetuple().tm_yday
        year_day.append(day)

    A['dia_año'] = year_day

    A['semana'] = (A['dia_año'] - 1) // 7 + 1

    return A

#%% 
def obtener_temperaturas_cercanas(longitud, latitud):
    # cargo históricos y Preparo los datos
    ds1 = xr.open_dataset('../datos/Datos_t2m_horario_2011a2021_uy.nc', chunks='auto')
    ds2 = xr.open_dataset('../datos/Datos_t2m_horario_2000a2010_uy.nc',chunks='auto')
    df1 = ds1.to_dataframe()
    df2 = ds2.to_dataframe()

    del ds1, ds2
    df3 = pd.concat([df2, df1], axis=0).reset_index()
    del df1, df2
    df3['t2m'] = df3['t2m'] - 273.15

    coordenadas = df3[['longitude', 'latitude']].drop_duplicates()

    distancias = [math.dist(coordenadas.iloc[i], [longitud, latitud]) for i in range(len(coordenadas))]
    coordenadas_cercanas = coordenadas.iloc[np.argmin(distancias)]

    df3_cercano = df3[(df3.longitude == coordenadas_cercanas[0]) & (df3.latitude == coordenadas_cercanas[1])].reset_index(drop = True)
    df3_cercano['time'] = pd.to_datetime(df3_cercano['time'])
    
    df3_cercano['Ano'] = df3_cercano['time'].dt.year 
    df3_cercano['Mes'] = df3_cercano['time'].dt.month 
    df3_cercano['DiaMes'] = df3_cercano['time'].dt.day  
    df3_cercano['Hora'] = df3_cercano['time'].dt.hour
    
    return df3_cercano.drop(columns=['longitude', 'latitude'])

# ------------------------------------------


#%% 
def generador_semanal(A, inicio, fin, n):

    """ A es el dataframe base que tenemos, los 22 años, ya filtrado por una latitud y longitud.
        inicio y fin son el año de inicio y el año de fin, respectivamente, del intervalo de años 
        que se desean utilizar para hacer el bootstrapp.
        n es el número de series sintéticas que se desean generar.
        Arg 
            A = DataFrame de temp (22 anos)
            inicio, fin periodo de tiempo apra evaluar
            n = cantidad de años sintéticos a generar
        Ret:
            df_esc_temp: lista de DataFrames que contiene las simulaciones para cada año sintetico
            
    """

    col = 'Ano_0'
    df_sem      = pd.DataFrame()        # almanceno la semana elegida al azar
    df_ano      = pd.DataFrame()        # Genero el ano sintetico
    df_esc_temp = pd.DataFrame()        # DataFrame que contiene todos los escenarios de temperatura

    for i in range(1,54):
        ano_rand = random.randint(inicio, fin)
        df_sem  = A[(A.Ano == ano_rand) & (A.semana == i)]
        df_ano = pd.concat([df_ano, df_sem], axis=0, ignore_index=True)
    
    df_esc_temp = df_ano
    df_esc_temp.rename(columns= {'t2m': col, 'DiaSemana': 'DiaSem_original'}, inplace=True)
    # Corrijo los dias d esemana para que tengan correlacion
    df_esc_temp['DiaSemana'] = range(8760)
    df_esc_temp['DiaSemana'] = (df_esc_temp['DiaSemana']//24)%7
    
    for ano in range(1,n):
        col = 'Ano_' + str(ano)
        df_sem      = pd.DataFrame()
        df_ano      = pd.DataFrame()
        for sem in range(1,54):
            ano_rand  = random.randint(inicio, fin)
            df_sem      = pd.DataFrame()
            df_sem[col] = A[(A.Ano == ano_rand) & (A.semana == sem)]['t2m']
            df_ano = pd.concat([df_ano, df_sem], axis=0, ignore_index=True)
        df_esc_temp[col] = df_ano
    # columnas = ['time', 'año', 'mes', 'dia', 'hora', 't2m', 'dia_año', 'semana']
    # columnas = ['time', 'DiaSemana','Mes','Dia','Hora', 't2m']
    
    return df_esc_temp


#%% 
def generar_datos_temp_sint(anos_to_sim =3, inicio =2001, fin =2021):
    
    # cargo históricos y Preparo los datos
    ds1 = xr.open_dataset('datos/Datos_t2m_horario_2011a2021_uy.nc', chunks='auto')
    ds2 = xr.open_dataset('datos/Datos_t2m_horario_2000a2010_uy.nc',chunks='auto')
    df1 = ds1.to_dataframe()
    df2 = ds2.to_dataframe()
    
    del ds1, ds2
    df3 = pd.concat([df2, df1], axis=0).reset_index()
    del df1, df2
    df3['t2m'] = df3['t2m'] - 273.15
    df3_carrasco = df3[(df3.longitude == -56) & (df3.latitude == -34.75)].reset_index(drop = True)
    del df3
    df3_carrasco = adecuar_dataframe(df3_carrasco)
    df_esc_temp = generador_semanal(df3_carrasco, inicio, fin, anos_to_sim)
    

    return df_esc_temp


#%% solo levanto los datos de alguna simulación
def cargar_datos_temp_sint(data_path= 'datos/temp_sintetico_1000.csv'):
    ##data_path      = 'data/temp_sintetico_1000.csv'
    
    try:
        data = pd.read_csv(data_path, engine='pyarrow')
    except FileNotFoundError:
        raise FileNotFoundError(f'Error: la ruta {data_path} para levantar los datos de Temperatura sintetica no fue encontrado.')
    
    
    return data