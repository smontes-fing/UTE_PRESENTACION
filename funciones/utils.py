# Funci√≥n de error (statistical measure)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import shutil
import matplotlib.pyplot as plt


# Constantes
Ener_2021 = 1.067156e+07


def r_squared(
    y_true,
    y_pred,
    ):
    """
    Calculates the R-squared value of two tensors.

    Args:
        y_true: Ground truth tensor.
        y_pred: Predicted tensor.

    Returns:
        R-squared value.
    """
    numerator = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    denominator = tf.reduce_mean(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.clip_by_value(tf.subtract(1.0, tf.math.divide(numerator, denominator)), clip_value_min=0.0, clip_value_max=1.0)
    return r2


def plot_densidades_escenarios(
    df_picos_demandas: pd.DataFrame, 
    df_picos_comparar: pd.DataFrame = pd.DataFrame({'Demanda' : []}), 
    df_picos_comparar2: pd.DataFrame = pd.DataFrame({'Demanda' : []}), 
    col1: str = 'Max_Pot',
    col2: str = 'Max_Pot',
    col3: str = 'Max_Pot',
    title: str = '',
    path: str = 'output/densidad_de_picos.png'
) -> None:
    """
    Plot the probability density of annual maxima of the given dataframes.

    Parameters
    ----------
    df_picos_demandas : pd.DataFrame
        The dataframe with the annual maxima of the scenario to be analyzed.
    df_picos_comparar : pd.DataFrame, optional
        The dataframe with the annual maxima of the scenario to be compared, by default empty.
    df_picos_comparar2 : pd.DataFrame, optional
        The dataframe with the annual maxima of the scenario to be compared, by default empty.
    col1 : str, optional
        The column name of the dataframe with the annual maxima of the scenario to be analyzed, by default 'Max_pot'.
    col2 : str, optional
        The column name of the dataframe with the annual maxima of the scenario to be compared, by default 'Max_pot'.
    col3 : str, optional
        The column name of the dataframe with the annual maxima of the scenario to be compared, by default 'Max_pot'.
    title : str, optional
        The title of the plot, by default ''.
    path : str, optional
        The path to save the plot, by default None.

    Returns
    -------
    None
    """
    # 3. Grafica la función de densidad de los picos de demanda
    plt.figure(figsize=(10, 6))

    if df_picos_comparar.empty: 
        df_picos_demandas[col1].plot(kind='density', color= 'r', linewidth=3)
        for cuantil in [5, 50, 95]:
            valor_cuantil = df_picos_demandas[col1].quantile(cuantil / 100)
            plt.axvline(x=valor_cuantil, color='r', linestyle='--', label=f'Cuantil {cuantil}')
    elif df_picos_comparar2.empty:
        df_picos_demandas[col1].plot(kind='density', color= 'r', linewidth=3)
        df_picos_comparar[col2].plot(kind='density', color= 'b',linewidth=3)
        for cuantil in [5, 50, 95]:
            valor_cuantil = df_picos_demandas[col1].quantile(cuantil / 100)
            plt.axvline(x=valor_cuantil, color='r', linestyle='--', label=f'Cuantil {cuantil}')
            valor_cuantil_comp = df_picos_comparar[col2].quantile(cuantil / 100)
            plt.axvline(x=valor_cuantil_comp, color='b', linestyle='--',label=f'Cuantil {cuantil}')
    else:
        df_picos_demandas[col1].plot(kind='density', color= 'r', linewidth=3)
        df_picos_comparar[col2].plot(kind='density', color= 'b',linewidth=3)
        df_picos_comparar2[col3].plot(kind='density', color= 'g',linewidth=3)
            
    plt.xlabel('Picos de Demanda (MW)')
    plt.ylabel('Densidad')
    if title == '':
        plt.title('Función de Densidad de Picos de Demanda Anuales')
    else: plt.title(title)
    plt.grid(True)
    plt.legend()
    if path != None:
        plt.savefig(path)

    plt.show()


def plot_POE(annmax: pd.Series,
             fig_path: str = 'output/POE.png') -> None:
    """Plot the Probability of Exceedance (PoE) of a given time series of annual maxima.

    The PoE is the likelihood that a maximum or minimum demand forecast will be met or exceeded.
    A 10% PoE maximum demand forecast, for example, is expected to be exceeded, on average, one year in 10,
    while a 90% PoE maximum demand forecast is expected to be exceeded nine years in 10.

    Parameters
    ----------
    annmax : pd.Series
        Time series of annual maxima.

    Returns
    -------
    None
    """
    # Define x and y variables
    xp = np.arange(0.01, 1.01, 0.01)
    yp = np.percentile(annmax, 100 * (1 - xp))

    # Set up the plot
    plt.plot(xp, yp,  label='PoE Demanda', color="steelblue", linewidth=2)
    ymax = max(yp)+100
    plt.vlines(x=0.05, 
               ymin=0 , ymax=ymax, color='r', linestyle='--',label='quantile 5%')
    plt.ylim(600, ymax)
    plt.xlim(0, 1.01)
    plt.xlabel("Probabilidad de Excedencia")
    plt.ylabel("Demanda (MW)")
    plt.title("")
    plt.legend()
    # Add a grid
    plt.grid(color="gray", linestyle="dotted")
    # Show the plot
    plt.savefig(fig_path)
    plt.show()


def crear_directorio(output_dir):
    """
    Crea el directorio de salida si no existe. Si existe, borra su contenido.

    Parameters:
        output_dir (str): Ruta del directorio de salida.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)


def graficar_entrenamiento(history, output_dir):
    """
    Genera y guarda las gráficas de resultados de entrenamiento.

    Parameters:
        history (History): Historial del entrenamiento.
        output_dir (str): Ruta del directorio donde guardar las gráficas.
    """
    crear_directorio(output_dir)
    
    accR2 = history.history['r_squared']
    val_accR2 = history.history['val_r_squared']
    accMAPE = history.history['mape']
    val_accMAPE = history.history['val_mape']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(accR2))

    # Graficar R² y pérdida
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accR2, label='Training R²')
    plt.plot(epochs_range, val_accR2, label='Validation R²')
    plt.legend(loc='lower right')
    plt.title('Training and Validation R²')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss (MAE)')
    plt.plot(epochs_range, val_loss, label='Validation Loss (MAE)')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss (MAE)')

    # Guardar la gráfica
    plt.savefig(os.path.join(output_dir, 'r_squared_and_loss.png'))
    plt.close()

    # Graficar MAPE y pérdida
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accMAPE, label='Training MAPE')
    plt.plot(epochs_range, val_accMAPE, label='Validation MAPE')
    plt.legend(loc='lower right')
    plt.title('Training and Validation MAPE')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss (MAE)')
    plt.plot(epochs_range, val_loss, label='Validation Loss (MAE)')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss (MAE)')

    # Guardar la gráfica
    plt.savefig(os.path.join(output_dir, 'mape_and_loss.png'))
    plt.close()

def validar_modelo(
    model: tf.keras.Sequential,  # Modelo a validar.
    input_data: pd.DataFrame,  # Datos de prueba.
    label_test: pd.DataFrame,  # Etiquetas reales de los datos de prueba.
    output_dir: str,  # Ruta del directorio donde guardar las gráficas.
    tipo: str  # Tipo de modelo (norm, OneHot, etc.).
) -> np.ndarray:
    """
    Valida el modelo contra datos de prueba y genera una gráfica comparativa.

    Parameters:
        model (tf.keras.Sequential): Modelo a validar.
        data_test (pd.DataFrame): Datos de prueba.
        label_test (pd.Series): Etiquetas reales de los datos de prueba.
        output_dir (str): Ruta del directorio donde guardar las gráficas.
        Ener_2021 (float): Escala energética de 2021.
        tipo (str): Tipo de modelo (norm, OneHot, etc.).

    Returns:
        np.ndarray: Predicciones del modelo.
    """

    # Evaluar el modelo
    print("Predicción del modelo con los datos de entrada de validación (2021)")
    pred = model.predict(input_data)
    print("Evaluo contra datos de validación (2021)")
    loss, acc, mape = model.evaluate(input_data, label_test)
    
    print(f"Loss = {loss}, acc = {acc}, mape = {mape}")

    # Guardar resultados de la evaluación
    with open(os.path.join(output_dir, f'resultados_{tipo}.txt'), 'w') as f:
        f.write(f"Modelo {tipo} validación: acc= {acc}, mape= {mape}, loss= {loss}\n")
    
    # comparo los valores de predicción y etiquetas reales
    if isinstance(label_test, pd.DataFrame):
        label_column = label_test.iloc[:, 0]  # Selecciona la primera columna
        label_subset = label_column.iloc[500:500+168]
    else:
        label_subset = label_test.iloc[500:500+168]

    pred_flat = pred.flatten()  # Aplana el array para convertirlo en una dimensión
    pred_series = pd.Series(pred_flat).reset_index(drop=True)
    pred_subset = pred_series.iloc[500:500+168]


    plt.figure(figsize=(12,6))
    label_subset.plot(label='Etiquetas reales', linestyle='-', color='blue')
    pred_subset.plot(label='Predicciones', linestyle='--', color='red')
    plt.xlabel('Índice')
    plt.ylabel('Demanda energía (MWh)')
    plt.title('Comparativa entre predicciones y etiquetas reales 2021')
    plt.legend()

    # Guardar la gráfica en un archivo
    plt.savefig(os.path.join(output_dir, 'validacion_2021.png'), dpi=300)
    plt.show()
    return pred

def graficar_predicciones(
    pred_norm: np.ndarray,  # Predicciones del modelo norm
    label_real: np.ndarray,  # Valores reales
    output_dir: str,  # Ruta del directorio donde guardar la gráfica
    ini: int = 180,  # Índice de inicio para la gráfica
    timeslots: int = 168  # Número de intervalos de tiempo a graficar
) -> None:
    """
    Genera una gráfica comparativa entre las predicciones de diferentes modelos y los valores reales.

    Parameters:
        pred_norm (ndarray): Predicciones del modelo norm.
        pred_oneHot (ndarray): Predicciones del modelo OneHot.
        pred_ANN (ndarray): Predicciones del modelo tradicional.
        label_real (ndarray): Valores reales.
        output_dir (str): Ruta del directorio donde guardar la gráfica.
        ini (int): Índice de inicio para la gráfica.
        timeslots (int): Número de intervalos de tiempo a graficar.
    """
   
    if ini < 0 or timeslots < 0:
        raise ValueError("No se permiten valores negativos")

    if ini + timeslots > len(pred_norm) or  ini + timeslots > len(label_real):
        raise ValueError("No se permiten índices fuera de rango")

    plt.figure(figsize=(10, 6))
    plt.plot(range(timeslots), pred_norm[ini: ini+timeslots], label='Predict_norm')
    plt.plot(range(timeslots), label_real[ini: ini+timeslots], label='Real')
    plt.legend(loc='upper right')
    plt.title('Predicción del modelo Vs Demanda 2021')
    save_path = os.path.join(output_dir, 'comparacion_predicciones.png')
    plt.savefig(save_path)
    print(save_path)
    plt.close()

def print_and_save_model_details(
    model: tf.keras.Model,
    output_dir: str = "model_details"
) -> None:
    """
    Imprime la arquitectura y los hiperparámetros del modelo Keras, y los guarda en un archivo JSON y un gráfico PNG.

    Args:
        model (tf.keras.Model): El modelo de Keras a inspeccionar.
        output_dir (str): Directorio donde se guardarán los archivos de salida. Por defecto es 'model_details'.

    Returns:
        None
    """

    # Chequear si el modelo es nulo
    if model is None:
        print("Modelo es nulo, no se puede guardar la información del modelo.")
        return

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Resumen del modelo
    model_summary = []
    try:
        model.summary(print_fn=lambda x: model_summary.append(x))
    except Exception as e:
        print(f"Error al obtener el resumen del modelo: {e}")
    model_summary_str = "\n".join(model_summary)
    print(model_summary_str)

    # Guardar el resumen en un archivo de texto
    try:
        with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
            f.write(model_summary_str)
    except Exception as e:
        print(f"Error al guardar el resumen del modelo en un archivo de texto: {e}")

    # Detalles de las capas
    layer_details = {}
    for layer in model.layers:
        layer_details[layer.name] = {
            "Tipo de capa": layer.__class__.__name__,
            "Hiperparámetros": layer.get_config(),
            "Parámetros entrenables": layer.count_params()
        }
        print(f"Capa: {layer.name}")
        print(f"  Tipo de capa: {layer.__class__.__name__}")
        print(f"  Hiperparámetros: {layer.get_config()}")
        print(f"  Parámetros entrenables: {layer.count_params()}")
        print("------------------------------------------------------")

    # Guardar los detalles de las capas en un archivo JSON
    try:
        with open(os.path.join(output_dir, "layer_details.json"), "w") as json_file:
            json_file.dump(layer_details, json_file, indent=4)
    except Exception as e:
        print(f"Error al guardar los detalles de las capas en un archivo JSON: {e}")

    # Imprimir y guardar los hiperparámetros del optimizador
    optimizer = model.optimizer
    if optimizer is not None:
        optimizer_params = {
            "Tasa de aprendizaje": optimizer.learning_rate.numpy(),
            "Decremento de tasa de aprendizaje": optimizer.decay.numpy(),
            "Nombre del optimizador": optimizer._name
        }
        print(f"Tasa de aprendizaje: {optimizer_params['Tasa de aprendizaje']}")
        print(f"Decremento de tasa de aprendizaje: {optimizer_params['Decremento de tasa de aprendizaje']}")
        print(f"Nombre del optimizador: {optimizer_params['Nombre del optimizador']}")

        # Guardar los hiperparámetros del optimizador en un archivo JSON
        try:
            with open(os.path.join(output_dir, "optimizer_params.json"), "w") as json_file:
                json_file.dump(optimizer_params, json_file, indent=4)
        except Exception as e:
            print(f"Error al guardar los hiperparámetros del optimizador en un archivo JSON: {e}")

    # Guardar la arquitectura en un archivo JSON
    try:
        model_json = model.to_json()
        with open(os.path.join(output_dir, "model_architecture.json"), "w") as json_file:
            json_file.write(model_json)
    except Exception as e:
        print(f"Error al guardar la arquitectura del modelo en un archivo JSON: {e}")

    # Visualizar y guardar la arquitectura del modelo en un archivo PNG
    try:
        plot_model(model, to_file=os.path.join(output_dir, "model_architecture.png"), show_shapes=True, show_layer_names=True)
    except Exception as e:
        print(f"Error al visualizar y guardar la arquitectura del modelo en un archivo PNG: {e}")




def transform_period_index(calendario, fecha_inicio_str = "2021-01-01", fecha_fin_str = "2022-10-31"):
    '''
    Esta función devuelve el TS (hora) de inicio y fin 
    
    Parameters
    calendario : ['DiaSemana', 'Mes', 'Dia', 'Hora']
        DESCRIPTION DF. COLUMNS 
    fecha_inicio_str : TYPE, optional
        DESCRIPTION. The default is "2021-01-01".
    fecha_fin_str : TYPE, optional
        DESCRIPTION. The default is "2022-10-31".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    indice

    '''

    fecha_inicio = pd.to_datetime(fecha_inicio_str + " 00:00:00")
    fecha_fin = pd.to_datetime(fecha_fin_str + " 23:59:59")
    
    if fecha_inicio > fecha_fin:
        raise ValueError("La fecha de inicio debe ser menor o igual que la fecha de fin.")

    # Extrae el año, mes y día de la fecha de búsqueda
    _, mes_ini, dia_ini = map(int, fecha_inicio_str.split('-'))
    _, mes_fin, dia_fin = map(int, fecha_fin_str.split('-'))

    # Encuentra el índice correspondiente a la fecha de búsqueda
    filtro_fecha_ini = (calendario["Mes"] == mes_ini) & (calendario["Dia"] == dia_ini)
    indice_fecha_ini = calendario[filtro_fecha_ini].index
    
    filtro_fecha_fin = (calendario["Mes"] == mes_fin) & (calendario["Dia"] == dia_fin)
    indice_fecha_fin = calendario[filtro_fecha_fin].index

    if indice_fecha_ini.empty:
        print("No se encontraron entradas para la fecha inicio", fecha_inicio_str)    
    if indice_fecha_fin.empty:
        print("No se encontraron entradas para la fecha fin", fecha_fin_str)
    else:    
        return indice_fecha_ini[0], indice_fecha_fin[-1]

def plot_demanda_hh(data, 
                    fecha_inicio_str,
                    fecha_fin_str,
                    periodo= (744,935),
                    title= 'Plot de demanda',
                    columnas=['Pot_evs', 'Pot_bus', 'Pot_hev'],
                    ):
    """
    Esta función muestra en una gráfica de área las demandas apra los diferentes EVs y la demanda total

    Parameters
    ----------
    data : TYPE pandas DataFrame 
        DESCRIPTION.
        debe contener al menos las siguientes columnas: ['Ano_x', 'Pot_evs', 'Pot_bus', 'Pot_hev']
    Returns
    -------Grafica la curva de demanda en formato horario
    None.

    """
# Lista de columnas que deseas graficar
    EV_data= data[columnas]
    EV_data.iloc[periodo[0]:periodo[1]].plot.area()
    # Agrear etiquetas y título
    plt.xlabel('Periodo de tiempo que va de' +fecha_inicio_str + ' hasta '+ fecha_fin_str + ' en horas.')
    plt.ylabel('Demanda de Vehículos Eléctricas')
    plt.title(title)
    # Mostrar el gráfico
    plt.show()
    
    
    # columnas_a_graficar = ['Demanda', 'Pot_evs', 'Pot_bus', 'Pot_hev']
    # Cuando existan mas de un Driver
    # Filtra las columnas que contienen el substring "Pot"
    columnas_a_graficar= data.filter(like='Pot').columns.tolist()
    for columna in data.columns:
        if 'Ano' in columna:
            data.rename(columns={columna: 'Demanda'}, inplace=True)
    columnas_a_graficar.insert(0,'Demanda')
    
    # Verificar la existencia de las columnas antes de graficar
    columnas_existentes = [col for col in columnas_a_graficar if col in data.columns]
    
    if columnas_existentes:
        # Al menos una de las columnas existe
        data[columnas_existentes].plot.area()
    
        # Agregar etiquetas y título
        plt.xlabel('Horas')
        plt.ylabel('Demanda MW')
        plt.title('Demanda de potencia por driver em el periodo '+fecha_inicio_str + ' hasta '+ fecha_fin_str)
    
        # Mostrar el gráfico
        plt.show()
    else:
        print("Ninguna de las columnas especificadas existe en el DataFrame.")


def def_rango_fecha(ano= '2040'):
    fecha_inicio = ano+'-01-01 00:00:00'
    fecha_fin =    ano+'-12-31 23:00:00'
    rango_fechas_raw = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='h')
    # Verifico que no sea Biciesto
    rango_fechas = rango_fechas_raw[~((rango_fechas_raw.month == 2) & (rango_fechas_raw.day == 29))]
    return rango_fechas