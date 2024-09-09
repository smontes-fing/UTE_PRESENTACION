#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:32:11 2023

@author: seba
"""
#%% Librerias a importar
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import random

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import MeanMetricWrapper
from pickle import dump, load


from funciones.utils_temp import cargar_datos_temp_sint, generar_datos_temp_sint
from funciones.utils_dem  import cargar_datos_dem_sint, genero_ano_dem_sint
from funciones.utils import r_squared
from funciones.utils import *
from funciones.utils_driverEVs import normalize_charge_curve, plot_evs_perfil_norm, EV_consumption_Ener
from funciones.utils_driverEVs import EV_consumption_hh, transform_period_index, plot_demanda_hh
from funciones.utils_driverEVs import def_rango_fecha
#%% Defino Parámetros de simulacion
"""
Se genera una gráfica de densidad de distribución y un análisis estadístico de los picos 
para los N escenarios de temperatura/demanda simulados
"""
ano = 2021              # Año a estudio´
anos_to_sim = 1000
tipo_modelo = 'norm'
version = 3
iteracion = 200

#%% Cargo el modelo y proy Energía para el Año elegido

# Cargo las proyecciones econométricas
# ener= 1.067156e+07
pbi_proy = pd.read_csv('datos/proy_econometrica_PBI.csv', engine='pyarrow')
ener = pbi_proy.loc[pbi_proy['ANO'] == ano, 'Energia'].iloc[0]  # Paso a MWh


# PATH_MODEL  = 'modelos/UTE_ANN_V2'+tipo_modelo
PATH_MODEL  = 'modelos/UTE_ANN_V2'+tipo_modelo+str(iteracion)
custom_objects = {'r_squared': r_squared}
model = load_model(PATH_MODEL , custom_objects= custom_objects)

# ===================== GENERO LOS DATOS SINTETICOS =====================
#===================================
#%%===== Datos de Temp ===============
# cargo los datos de temperatura
temp_path = 'datos/Temp_sim/temp_sintetico_'+ str(anos_to_sim) +'.csv'
data_temp = cargar_datos_temp_sint(temp_path)

# Genero los datos de Temp
# data_temp = generar_datos_temp_sint(anos_to_sim) 
# data_temp.to_csv(temp_path , index = False)

#===================================
#%% Datos de Drivers EVs

# Curva de carga normalizada de diferente vehículos
tipo = 'BaU'
perfil_carga_norm = normalize_charge_curve(tipo)
plot_evs_perfil_norm(perfil_carga_norm)

# Defino cantidad de vehiculos por tipo [EVS, buses, camiones]
N_evs  = 100000
N_buses= 1000
N_heavy= 5600
EVs_dem= [N_evs,N_buses,N_heavy]
if ano == 2021: 
    EVs_dem= [0, 0, 0]
calendario = data_temp.loc[:, ['DiaSemana', 'Mes', 'Dia', 'Hora']]
# Consumo de los EVs
dem_driver_EVs = EV_consumption_hh(calendario, EVs_dem, perfil_carga_norm)
energia_Driver_Evs = EV_consumption_Ener(dem_driver_EVs)

print(f'Energía Total consumida por el Driver Transporte eléctrico: {energia_Driver_Evs}')
print(f'Porcentaje de energía del Driver conparado con la energia anual: {energia_Driver_Evs/ener}')


#%% Prediccion Demanda 
#===================== Modelo Buisness as Usual (BaU) ===============
# Cargo los datos de demanda
demanda_path = 'datos/demanda_sim/dem_sintetico_'+ str(anos_to_sim) + tipo_modelo +'_' + str(ano)+ '.csv'
# predicc= cargar_datos_temp_sint(demanda_path)

# Genero los datos de Demanda
predicc = genero_ano_dem_sint(data_temp, anos_to_sim, model, 
                              tipo_modelo, version, iteracion, ener)
predicc.to_csv(demanda_path , index = False)


#===================================
# DataFrame con las demandas de los diferentes escenarios y la demanda de los Drivers
predicc = pd.merge(predicc, dem_driver_EVs, left_index=True, right_index=True)
rango_fechas = def_rango_fecha(str(ano))
predicc.index = rango_fechas

# Filtrar el DataFrame para la semana específica
fecha_inicio_semana = str(ano)+'-06-15'
fecha_fin_semana = pd.Timestamp(fecha_inicio_semana) + pd.DateOffset(days=7)  # Agregar 6 días para obtener una semana completa

ano_to_show  = 'Ano_' + str(random.randint(0, anos_to_sim-1))
# ano_to_show  = 'Ano_1'

data_to_show = predicc[[ano_to_show, 'Pot_evs', 'Pot_bus', 'Pot_hev']]
datos_semana = data_to_show.loc[fecha_inicio_semana:fecha_fin_semana]
# Trazar el gráfico de áreas apiladas
datos_semana.plot(kind='area', stacked=True, figsize=(10, 6))
# Configurar etiquetas y título
plt.xlabel('Hora del día')
plt.ylabel('Potencia')
plt.ylim(0, 3000)
plt.savefig('Figuras/proy_EV_main.png')
plt.title('Proyeccion de penetración alta y carga BaU- Semana del ' + fecha_inicio_semana + ' al ' + fecha_fin_semana.strftime('%Y-%m-%d'))
# Mostrar la leyenda
plt.legend()
# Mostrar el gráfico
plt.show()

# ================================================================
#%%  ANALISIS ESTADISTICO 
# Estadisticas de los n anos simulados
estadisticas = pd.DataFrame()
data =         pd.DataFrame()
for ano in range(anos_to_sim):
    col = 'Ano_' + str(ano)
    demanda = predicc[col] + predicc['Pot_evs'] + predicc['Pot_bus'] + predicc['Pot_hev']
    estadisticas.at[ano,'mean']    = demanda.mean()
    estadisticas.at[ano,'Max_Pot'] = demanda.max()
    estadisticas.at[ano,'Energia'] = demanda.sum()


# ================================================================
#%% Plots de Demanda

# Plot de la funcion de densidad de picos
plot_densidades_escenarios(estadisticas)

# Funcion de probabilidad de excedencia PoE 
plot_POE(predicc[ano_to_show])






