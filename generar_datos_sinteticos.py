import pandas as pd
import xarray as xr
import random
import argparse
import netCDF4
import dask
import time as tm

import warnings
warnings.filterwarnings("ignore", message="The behavior of .*")

def load_data_CDF4(file_paths):
    """
    Carga y combina los datos desde múltiples archivos.

    Args:
    file_paths (list of str): Lista de rutas de archivos a cargar.

    Returns:
    pd.DataFrame: Datos combinados en un DataFrame de pandas.
    """
    start_time = tm.time()
    dfs = []
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

def load_data_csv(file_paths):
    """
    Carga y combina los datos desde múltiples archivos CSV.

    Args:
    file_paths (list of str): Lista de rutas de archivos a cargar.

    Returns:
    pd.DataFrame: Datos combinados en un DataFrame de pandas.
    """
    start_time = tm.time()
    
    dfs = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
        except Exception as e:
            raise FileNotFoundError(f"Error al cargar el archivo {file_path}: {e}")
    
    end_time = tm.time()
    # Medir el tiempo de carga
    load_time = end_time - start_time
    print(f"Tiempo de carga del archivo: {load_time:.2f} segundos")

    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined_df = combined_df.rename(columns={'temperatura(°C)': 't2m'})    
    # Filtrar las entradas donde longitud es igual a -56.00
    df_filtrado = combined_df[combined_df['longitud'] == -56.00]

    return df_filtrado

def adecuar_dataframe(df):
    """
    Prepara el DataFrame separando la fecha en año, mes, día y hora, y agrega columnas adicionales.

    Args:
    df (pd.DataFrame): DataFrame original.

    Returns:
    pd.DataFrame: DataFrame modificado.
    """
    # Convertir toda la columna de tiempo a datetime
    df['time'] = pd.to_datetime(df['time'])

    # Extraer partes de la fecha de una vez utilizando atributos de datetime
    df['Ano'] = df['time'].dt.year
    df['Mes'] = df['time'].dt.month
    df['Dia'] = df['time'].dt.day
    df['Hora'] = df['time'].dt.hour
    df['DiaSemana'] = df['time'].dt.dayofweek
    df['dia_año'] = df['time'].dt.dayofyear
    df['semana'] = (df['dia_año'] - 1) // 7 + 1

    return df[['time', 'DiaSemana', 'Ano', 'Mes', 'Dia', 'Hora', 't2m', 'semana']]

def adecuar_dataframe_csv(df):
    """
    Prepara el DataFrame separando la fecha en año, mes, día y hora, y agrega columnas adicionales.

    Args:
    df (pd.DataFrame): DataFrame original.

    Returns:
    pd.DataFrame: DataFrame modificado.
    """
    # Renombrar las columnas para tener nombres uniformes
    df = df.rename(columns={'año': 'Ano', 'mes': 'Mes', 'día': 'Dia', 'hora': 'Hora', 'temperatura(°C)': 't2m'})
    
    # Convertir las columnas de fecha y hora a enteros
    df['Ano'] = df['Ano'].astype(int)
    df['Mes'] = df['Mes'].astype(int)
    df['Dia'] = df['Dia'].astype(int)
    df['Hora'] = df['Hora'].astype(int)

    # Crear la columna 'time' a partir de las columnas de fecha y hora existentes
    df['time'] = pd.to_datetime(
        df['Ano'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2) + '-' + 
        df['Dia'].astype(str).str.zfill(2) + ' ' + df['Hora'].astype(str).str.zfill(2) + ':00:00'
    )

    # Extraer partes de la fecha de una vez utilizando atributos de datetime
    df['DiaSemana'] = df['time'].dt.dayofweek
    df['dia_año'] = df['time'].dt.dayofyear
    df['semana'] = (df['dia_año'] - 1) // 7 + 1

    # Imprimir información de depuración después de modificar el DataFrame

    return df[['time', 'DiaSemana', 'Ano', 'Mes', 'Dia', 'Hora', 't2m', 'semana']]

def generador_semanal(A, inicio, fin, n):
    lista_dataframes = []
    print(f"Años para sortear semanas {inicio =}, {fin =}")
    for j in range(1, n + 1):
        df_sintetic = pd.DataFrame(columns=A.columns)
        
        for i in range(1, 54):  # Asegura que se recorran todas las semanas (1 a 52) + sem 53 dia 31/12
            ano_rand = random.randint(inicio, fin)
            df_bloque = A[(A.Ano == ano_rand) & (A.semana == i)]
            
            if not df_bloque.empty:  # Verificar si df_bloque tiene filas antes de concatenar
                df_sintetic = pd.concat([df_sintetic, df_bloque], axis=0, ignore_index=True)

        df_sintetic = df_sintetic.rename(columns={'t2m': 'Temp'})
        df_sintetic['sintetic'] = j
        lista_dataframes.append(df_sintetic)
    
    # Concatenar todos los años sintéticos en un solo DataFrame
    df_final = pd.concat(lista_dataframes, ignore_index=True)
    return df_final


def main(years_to_generate, output_path, input_paths):
    """
    Función principal para ejecutar la generación de series sintéticas.

    Args:
    years_to_generate (int): Número de años a generar.
    output_path (str): Ruta del archivo de resultados.
    input_paths (str): Rutas de archivos de datos de entrada, separadas por comas.
    """

    if input_paths:
        
        input_paths_list = input_paths.split(',')
    else:
        input_paths_list = ['../UTE_E2/datos/dataset_2011a2021_TEMP.csv', '../UTE_E2/datos/dataset_2000a2010_TEMP.csv']

    if not output_path:
        output_path = '../UTE_E2/datos/temp_sintetico_{}.csv'.format(years_to_generate)

    try:
        data = load_data_csv(input_paths_list)
    except FileNotFoundError as e:
        print(e)
        return
    
#    data = adecuar_dataframe(data)
    data = adecuar_dataframe_csv(data)

    inicio = data['Ano'].min()
    fin = data['Ano'].max()
    df_simulacion = generador_semanal(data, inicio, fin, years_to_generate)

    try:
        df_simulacion.to_csv(output_path, index=False)
        print(f"Archivo guardado en {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo en {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de series sintéticas de temperatura.')
    parser.add_argument('--years', type=int, required=True, help='Número de años a generar')
    parser.add_argument('--output', type=str, help='Ruta del archivo de resultados')
    parser.add_argument('--inputs', type=str, help='Rutas de archivos de datos de entrada, separadas por comas')
    args = parser.parse_args()

    try:
        main(args.years, args.output, args.inputs)
    except Exception as e:
        print(f"Error: {e}")
