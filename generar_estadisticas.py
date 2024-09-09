#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:32:11 2023

@author: seba
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import argparse
from icecream import ic
from funciones.utils_temp import cargar_datos_temp_sint
from funciones.utils_dem import cargar_datos_dem_sint 
from funciones.utils_driverEVs import (
    cargar_datos_EV, 
    graficar_curvas_carga_normalizadas, 
    EV_consumption_Ener, 
    EV_consumption_hh, 
    probabilidad_carga_EVs
)
from funciones.utils import (
    def_rango_fecha,
    plot_POE,
)

def main(
    year: int,
    demand_path: str,
    charge_curve: str,
    ev_list: list,
) -> None:
    """
    Main function to generate the data for the UTE case study.

    Parameters
    ----------
    year : int
        The year to generate the data for.
    demand_path : str
        The path to the demand data file.
    charge_curve : str
        The type of charge curve to use.
    ev_list : list
        The list of EVs to generate data for.

    Returns
    -------
    None
    """
    # Carga datos de temperatura
    temp_data_path = 'datos/temp_sintetico_2.csv'
    temp_data = cargar_datos_temp_sint(temp_data_path)
    calendar = temp_data[['DiaSemana', 'Mes', 'Dia', 'Hora']]

    # Carga la proyecctión de energía segun PBI hasta 2050
    pbi_proyection_path = 'datos/proy_econometrica_PBI.csv'
    pbi_proy = pd.read_csv(pbi_proyection_path, engine='pyarrow')
    energy_proy = pbi_proy.loc[pbi_proy['ANO'] == year, 'Energia'].iloc[0]
    energy_base = pbi_proy.loc[pbi_proy['ANO'] == 2021, 'Energia'].iloc[0]
    print(f"proyección de energía segun PBI para {year} MWh", energy_proy)
    print(f"proyección de energía segun PBI para 2021 MWh", energy_base)

    # Carga Los datos de demanda simulada
    demand_data = cargar_datos_dem_sint(
                                        demand_path, 
                                        energy_proy,
                                        energy_base
                                        )
    _, num_years = demand_data.shape

    # Genera los datos del Driver EV
    ev_data, _ = proyeccion_driver_EV(year, ev_list, charge_curve, calendar)

    # Merge de los dos dataframes demand_data y ev_data
    merged_data = pd.merge(demand_data, ev_data, left_index=True, right_index=True)

    # Muestro una sema especifica del anio con la demanda driver y la proyectada
    # Preprocess de los datos de calendario
    date_range = def_rango_fecha(str(year))
    merged_data.index = date_range
    start_date = f'{year}-06-15'
    end_date = pd.Timestamp(start_date) + pd.DateOffset(days=7)
    if 'Ano' in merged_data.columns:
        year_to_show = f'Ano_{random.randint(0, num_years - 1)}'
    else:
        year_to_show = f'{random.randint(0, num_years - 1)}'
    data_to_show = merged_data[[year_to_show, 'Pot_evs', 'Pot_bus', 'Pot_hev']]
    week_data = data_to_show.loc[start_date:end_date]
    plot_datos_semana(week_data, start_date, end_date)

    # Analisis de resultados

    plot_POE(merged_data[year_to_show]) # probabildiad de excedencia
   
    

    # return diccionario con las estadísticas para las curvas BaU y ToU


def proyeccion_driver_EV(
    ano: int,
    lista_evs: list,
    curva = str,
    calendario = pd.DataFrame,
):
    # Normalizar la curva de carga de vehículos eléctricos
    perfil_carga_norm = probabilidad_carga_EVs(curva)
    path_figure = f"output/probabilidad_carga_{curva}.png"
    graficar_curvas_carga_normalizadas(perfil_carga_norm, path_figure)

    if not lista_evs:
        EVs_estimada = cargar_datos_EV(ano)
    elif len(lista_evs) == 1:
        lista_evs.append([0, 0])
        EVs_estimada = lista_evs
    else:
        raise ValueError("La lista de EVs debe tener 0 o 1 elementos")
    
    # Definir cantidad de vehículos por tipo y calcular el consumo
    dem_driver_EVs = EV_consumption_hh(calendario, EVs_estimada, perfil_carga_norm)
    energia_Driver_Evs = EV_consumption_Ener(dem_driver_EVs)

    return dem_driver_EVs, energia_Driver_Evs

def plot_datos_semana(
    datos_semana: pd.DataFrame, 
    fecha_inicio: str, 
    fecha_fin: pd.Timestamp
) -> None:
    """
    Plot the data for a specific week.

    Parameters
    ----------
    datos_semana : pd.DataFrame
        The data to plot.
    fecha_inicio : str
        The start date of the week.
    fecha_fin : pd.Timestamp
        The end date of the week.

    Returns
    -------
    None
    """

    datos_semana.plot(kind='area', stacked=True, figsize=(10, 6))
    plt.xlabel('Hora del día')
    plt.ylabel('Potencia')
    plt.ylim(0, 3000)
    plt.title(f'Proyección de penetración alta y carga BaU- Semana del {fecha_inicio} al {fecha_fin.strftime("%Y-%m-%d")}')
    plt.legend()
    plt.savefig('output/proy_EV_main.png')
    plt.show()

def analizar_demanda(predicc, anos_to_sim):
    
    estadisticas = pd.DataFrame()
    for ano in range(anos_to_sim-1):
        if 'Ano' in predicc.columns:
            col = f'Ano_{ano}'
        else:
            col = str(ano)
        demanda = predicc[col] + predicc['Pot_evs'] + predicc['Pot_bus'] + predicc['Pot_hev']
        estadisticas.at[ano, 'mean'] = demanda.mean()
        estadisticas.at[ano, 'Max_Pot'] = demanda.max()
        estadisticas.at[ano, 'Energia'] = demanda.sum()
    return estadisticas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Análisis de predicciones de demanda de energía.')
    parser.add_argument(
                        '--ano',        
                        type=int,   
                        default= 2021,
                        help='Año de predecir'
                        )
    parser.add_argument(
                        '--demanda_path', 
                        type=str,   
                        default='datos/dem_sintetico_100.csv', 
                        help='File_path del archivo de predicciones, default: datos/dem_sintetico_100.csv'
                        )
    parser.add_argument('--curva',      
                        type= str,  
                        default='BaU', 
                        help='Tipo de curva de carga, defalt: BaU'
                        )
    parser.add_argument('--evs',        
                        type= list, 
                        default=[], 
                        help='Penetración de EVs, defalt: []'
                        )
    args = parser.parse_args()
    
    # cProfile.run('main(args.ano, args.input_path, args.curva, args.evs)')
    main(args.ano, args.demanda_path, args.curva, args.evs)