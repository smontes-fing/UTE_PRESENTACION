#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:51:58 2023

TYNDP 2020 Scenario Building Guidelines - Report jun 2020 sección 5.2.3 EV Load Patterns 
Este escenario crea un modelo de consumo para vehículos Eléctricos, diferenciando entre tres tipos: EV light-duty,
Heavy-duty (Camiones de transporte no utilitarios) y buses eléctricos.

Se generan 3 perfiles normalizados de consumo de distintos grupos de vehículos eléctricos: light-duty, buses, Heavy-duty
para dos escenarios diferentes (tarifa plana, es decir cargar cuando llegan a casa, y charging ToU tariff).
@author: seba
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from funciones.utils import (
    transform_period_index, 
    plot_demanda_hh,
    def_rango_fecha
)


# Perfiles de carga de EVs (Modelos Bottom-up)
def cargar_datos_EV(año_proyeccion):
    if año_proyeccion < 2021 or año_proyeccion > 2050:
        raise ValueError("El año de proyección debe estar entre 2021 y 2050.")

    # Define the years and data for interpolation
    years = [2021, 2030, 2040]
    data = np.array([
        [0, 0, 0],  # [Evs, Buses, Heavy_EV]
        [70000, 710, 4000],
        [100000, 1000, 5600]
    ])

    # Create linear interpolator
    f = interp1d(years, data, axis=0, kind='linear')

    # Interpolate for the given projection year
    interpolated_data = f(año_proyeccion)

    return interpolated_data

def probabilidad_carga_EVs(tipo= 'BaU'):

    # The daily load profile was created based on the assumption that most of buses uses slow night
    # charging. Fast chargers at bus depots are also available for around 30-35 % of the fleet. This
    # profile reflects mainly public transport in cities.
    if tipo == 'BaU':
        charge_evs = np.array([
                            0.105, 0.105, 0.105, 0.08, 0.05, 0.03, 0.01, 0.01, 
                            0.01, 0.03, 0.03, 0.03, .025, .025, .025, 0.01, 
                            0.01, 0.01, .020, 0.04, .045, .045, .045, .105
                               ])
                               

        charge_bus = np.array([
                            .089, .086, .086, .082, .063, .060, .014, .019, 
                            .019, .019, .018, .018, .017, .017, .017, .017,
                            .021, .024, .024, .033, .037, .048, .086, .086
                            ])

        charge_hev = np.array([
                            .11, 0.11, 0.11, 0.1, 0.1, 0.02, .005, .005, 
                            .005, .005, .005, .005, .005, .005, .005, .005, 
                            .005, .005, .005, 0.02, 0.055, .09, 0.11, 0.11
                               ])

    elif tipo == 'ToU':
        charge_evs = np.array([
                            0.0, 0.00, 0.4, 0.3, 0.3, 0.00, .0, 
                            0.0, 0.00, 0.0, 0.0, 0.0, .0, .0, .0,
                            0.0, .0, .0, .0, .0, .0, .0, .0, .0
                                 ])

        charge_bus = np.array([
                            .089, .086, .086, .082, .063, .060, .014, .019,
                            .019, .019, .018, .018, .017, .017, .017, .017, 
                            .021, .024, .024, .033, .037, .048, .086, .086
                            ])

        charge_hev = np.array([
                            .11, 0.11, 0.11, 0.1, 0.1, 0.02, .005, .005, .005,
                            .005, .005, .005, .005, .005, .005, .005, .005,
                            .005, .005, 0.02, 0.055, .09, 0.11, 0.11
                            ])

    else:
        error_message = f'Tipo de curva "{tipo}" no definido'
        raise ValueError(error_message)

    return [charge_evs, charge_bus, charge_hev]
    
def graficar_curvas_carga_normalizadas(curvas_carga_norm, 
                                       ruta = None):
    '''
    Función que grafica las curvas normalizadas de carga de los tipos de vehículos eléctricos
    Parámetros
    curvas_carga_norm : List
        Descripción.
        lista de curvas de carga
    Retorna
    None.
    '''
    evs = curvas_carga_norm[0]
    buses = curvas_carga_norm[1]
    hev = curvas_carga_norm[2]

    plt.plot(buses, label='Autobuses')
    plt.plot(hev, label='Vehículos eléctricos pesados')
    plt.plot(evs, label='Vehículos eléctricos')
    plt.legend()
    plt.grid()
    plt.ylabel("Probabilidad %")
    plt.xlabel("Hora del día")
    plt.title("Curva de probabilidad de carga")
    if ruta != None:
        plt.savefig(ruta)
    plt.show(block=False)

def EV_consumption_hh(period, EVs, perfiles,
                   consumo = [5,190,100], # kWh/dia
                   ):
    """ 
    Esta funcion clcula la demanda de los diferentes EV, tomando como entrada 3 tipos de EVs,
    su consumo medio diario y la curva normalizada de carga
    Args:
            period: periodo de tiempo (horario) para el cual calcular el consumo
                DataFrame("mes","dia", "hora", "dia_sem")
            evs: cantidad esperada de vehículos livianos
                int
            buses: cantidad esperada de buses eléctricos
                int
            heavy_ev: cantidad esperada de camiones y vehículos pesados
                int
            consumption: consumo medio kW/dia- datos Observatorio Movilidad 2023 https://montevideo.gub.uy/observatorio-de-movilidad
                list[int] = [con_evs, con_bus, con_hev] = 
                            consumo = kwh/km * km/dia
                            autos privados = 0.15 kWh/km * 20 km/dia
                            bus = 1.11 kWh/km * 170 km/dia
                            camiones =  0.5 kWh/km * 100 km/dia
            Effective usage:    the average use of a specified vehicle type, divided into weekdays and
                                weekends (each one with a chosen charge profile), expressed in km per day
                list[int] = [weekday, weekend]
            perfiles = perfiles de consumo normalizados para cada tipo de trasnporte
                list[np.array(24)] = [BaU-evs, BaU_bus, BaU_hev]
        
        Returns perfil de consumo
    """
    evs     = EVs[0] 
    buses   = EVs[1]
    heavy_evs= EVs[2]
    
    cons= pd.DataFrame()
    for indice in period.index:
        cons.at[indice, 'Pot_evs'] = perfiles[0][period['Hora'].iloc[indice]]*evs*consumo[0]/1000    # Paso a MW para ser consistente con Demanda
        cons.at[indice, 'Pot_bus'] = perfiles[1][period['Hora'].iloc[indice]]*buses*consumo[1]/1000
        cons.at[indice, 'Pot_hev'] = perfiles[2][period['Hora'].iloc[indice]]*heavy_evs*consumo[2]/1000
        
        
    return cons

# ==============================================================================================
#%% Cálculo de energía de EVs

def EV_consumption_Ener(consumo):
    """ 
    Esta funcion calcula el consumo de energia del parque vehícular de EV en el periodo en formato horario.
    Args
            consumo: consumo en un horizonte de tiempo
            consumo = list("mes","dia", "hora", "dia_sem", "demanda", Pot_evs, Pot-bus, Pot_hev)
        Returns: 
            Energía total consumida por el parque vehicular en el periodo
        """

    e_evs = consumo["Pot_evs"].sum()
    print(f'Energía consumida por light-duty evs: {e_evs}')
    bus   = consumo["Pot_bus"].sum()
    print(f'Energía consumida por buses eléctricos: {bus}')
    H_evs = consumo["Pot_hev"].sum()
    print(f'Energía consumida por Heavy-duty evs: {H_evs}')
    energia = e_evs + bus + H_evs
    return energia

def calc_driver(data_temp, 
                EVs_dem,
                tipo = 'BaU',
                ):

    # Curva de carga normalizada de diferente vehículos
    perfil_carga_norm = probabilidad_carga_EVs(tipo)

    # Consumo de los EVs
    calendario = data_temp.loc[:, ['DiaSemana', 'Mes', 'Dia', 'Hora']]
    dem_driver_EVs = EV_consumption_hh(calendario, EVs_dem, perfil_carga_norm)
    energia_Driver_Evs = EV_consumption_Ener(dem_driver_EVs)
    
    return dem_driver_EVs, energia_Driver_Evs 
    
#
def dem_EVs(dem):
    '''
    Calcula la demanda de energía horaria para el driver EV

    Parameters
    ----------
    dem : TYPE pandas DataFrame
        DESCRIPTION.
        COntiene la demanda horaria anual de los EVs por tipo de EVs
    Returns
    -------
    dem_EV : TYPE List
        DESCRIPTION.
        Demanda horaria del driver EVs
    '''
    dem_EV = dem['Pot_evs'].values + dem['Pot_bus'].values + dem['Pot_hev'].values
    return dem_EV

