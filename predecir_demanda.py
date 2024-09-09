import pandas as pd
import argparse
import os
from icecream import ic 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pickle import load
from funciones.utils_transform import (
    transform,
    transform_norm,
    transform_oneHot,
)
from generar_estadisticas import plot_datos_semana
from funciones.utils import def_rango_fecha

def load_data(input_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Parameters:
    input_path (str): Ruta del archivo CSV.
    
    Returns:
    pd.DataFrame: Datos cargados en un DataFrame de pandas.
    """
    try:
        data = pd.read_csv(input_path, engine='pyarrow')
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar los datos desde {input_path}: {e}")


def preprocess_data(
    data: pd.DataFrame,
    tipo: str = 'norm',
    one_hot_columns: list[str] = ['Mes', 'Dia', 'DiaSemana', 'Hora'], 
    Key_Label: str = 'Demanda', 
    columnas_feature: list[str] = ['DiaSemana', 'Mes', 'Dia', 'Hora','Temp', 'Demanda_ener'], 
    e_base: int = 0, 
    ) -> tuple[pd.DataFrame, pd.Series, list[str], int]:
    """
    Preprocesa las features de los datos para la predicción.
    
    Args:
    data (pd.DataFrame): Datos Temp.
    tipo (str, optional): Tipo de preprocesamiento. Defaults to 'norm'.
    one_hot_columns (list[str], optional): Columnas a codificar con one-hot. Defaults to ['Mes', 'Dia', 'DiaSemana', 'Hora'].
    Key_Label (str, optional): Columna con la etiqueta. Defaults to 'Demanda'.
    columnas_feature (list[str], optional): Columnas a utilizar como features. Defaults to ['DiaSemana', 'Mes', 'Dia', 'Hora','Temp', 'Demanda_ener'].
    e_base (int, optional): Valor de la energía base. Defaults to 0.
    
    Returns:
    pd.DataFrame: Datos preprocesados listos para la predicción.
    pd.Series: Etiquetas asociadas a los datos preprocesados.
    list[str]: Columnas features resultantes.
    int: Cantidad de features resultantes.
    """
    train= False
    ic("Datos de Temperatura",data.columns)
    if tipo == 'norm': 
        data_in, label_in, columnas_fature, features_size = transform_norm(data, train, Key_Label, columnas_feature, e_base)
    elif tipo == 'OneHot':
        data_in, label_in, columnas_fature, features_size = transform_oneHot(data, train, one_hot_columns, Key_Label, columnas_feature)
    elif tipo == '':
        data_in, label_in, columnas_fature, features_size = transform(data, train, Key_Label, columnas_feature)
    else:
        raise ValueError(f'Tipo desconocido: {tipo}')

    ic(columnas_fature)

    return data_in, label_in, columnas_fature, features_size


def r_squared(
    y_true: tf.Tensor,  # Valores reales.
    y_pred: tf.Tensor   # Valores predichos.
) -> tf.Tensor:  # Valor de R^2.
    """
    Calcula el coeficiente de determinación (R^2) entre los valores reales y predichos.
    
    Parameters:
        y_true (tf.Tensor): Valores reales.
        y_pred (tf.Tensor): Valores predichos.
    
    Returns:
        tf.Tensor: Valor de R^2.
    """
    numerator = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
    denominator = tf.reduce_mean(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.clip_by_value(tf.subtract(1.0, tf.math.divide(numerator, denominator)), clip_value_min=0.0, clip_value_max=1.0)
    return r2

def load_model_and_scaler(
    model_path: str,  # Path to the saved model file.
    scaler_path: str  # Path to the saved scaler file.
) -> tuple[tf.keras.Model, MinMaxScaler]:  # Loaded model and scaler.
    """
    Loads the model and scaler from the given paths.
    
    Args:
        model_path (str): Path to the saved model file.
        scaler_path (str): Path to the saved scaler file.
    
    Returns:
        Tuple[tf.keras.Model, MinMaxScaler]: Loaded model and scaler.
    """
    custom_objects = {'r_squared': r_squared}
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar el modelo desde {model_path}: {e} \n")

    try:
        scaler = load(open(scaler_path, 'rb'))
    except Exception as e:
        raise FileNotFoundError(f"Error al cargar el escalador desde {scaler_path}: {e} \n")
    ic(type(model))
    ic(type(scaler))
    return model, scaler

def fijar_energia_anual(
    ano: int  # A o para el que se va a predecir la energ a.
) -> tuple[float, float]:  # Valor de la energ a objetivo y base.
    """
    Fija la energ a objetivo y base para el a o dado.
    
    Args:
        ano (int): A o para el que se va a predecir la energ a.
    
    Returns:
        Tuple[float, float]: Valor de la energ a objetivo y base.
    """
    # e_obj= 1.067156e+07
    econom_proj= pd.read_csv('datos/proy_econometrica_PBI.csv', engine='pyarrow')
    # Energ a base (2021)
    e_base  = econom_proj.loc[econom_proj['ANO'] == 2021, 'Energia'].iloc[0]  # Paso a MWh
    # Energ a objetivo (a o a predecir)
    e_obj   = econom_proj.loc[econom_proj['ANO'] == ano , 'Energia'].iloc[0]  # Paso a MWh

    return e_obj, e_base

def generar_predicciones(
    data_temp: pd.DataFrame,
    e_obj: float,
    e_base: float,
    model_type: str,
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    output_path: str
) -> pd.DataFrame:
    """
    Generates predictions and saves them to a CSV file.
    Genera las predicciones y guarda los resutlados en un archivo CSV file
    Args:
        data_temp (pd.DataFrame): DataFrame with the data to be processed.
        e_obj (float): Energy value for the object.
        e_base (float): Energy base value.
        model_type (str): Type of the model.
        model (tf.keras.Model): Loaded model.
        scaler (MinMaxScaler): Loaded scaler.
        output_path (str): Path to the output file.

    Returns:
        pd.DataFrame: DataFrame with the predictions.
    """
    df_pred = pd.DataFrame()
    anos_to_sim = len(data_temp['sintetic'].unique())

    for ano in range(anos_to_sim):
        col = 'Ano_' + str(ano)
        data_raw = pd.DataFrame()
        # columnas= ['DiaSemana', 'Mes', 'Dia', 'Hora', col, 'Demanda_ener']
        columnas= ['DiaSemana', 'Mes', 'Dia', 'Hora', 'Temp']
        data_raw = data_temp[columnas]
        ic(data_raw.head(3))
        data_in, _, _, _ = preprocess_data(data_raw, 
                                       tipo=model_type, 
                                       one_hot_columns=['Mes', 'Dia', 'DiaSemana', 'Hora'], 
                                       Key_Label='Demanda', 
                                       columnas_feature= columnas, 
                                       e_base=e_obj, 
                                       # Codigo_estacion='',
                                       )
        
        data_in = scaler.transform(data_in)
        df_pred[col]=pd.DataFrame(model.predict(
                              x=data_in,
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
        if model_type != '':
            df_pred[col] = df_pred[col] * (e_obj/e_base)
    try:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        df_pred.to_csv(output_path, mode='w', index=False)
    except Exception as e:
        raise IOError(f"Error al guardar las predicciones en {output_path}: {e}")
    
    return df_pred

def main(
    ano_to_predict: int,  # Año a predecir
    input_path: str,  # Ruta del archivo de datos de entrada
    model_type: str,  # Tipo de modelo
    output_path: str,  # Ruta del archivo de resultados
    model_path: str,  # Ruta del archivo del modelo
    scaler_path: str  # Ruta del archivo del escalador
) -> None:
    """
    Función principal para ejecutar el flujo completo de carga, preprocesamiento, predicción y guardado de resultados.
    
    Args:
    ano_to_predict (int): Año a predecir
    input_path (str): Ruta del archivo de datos de entrada
    model_type (str): Tipo de modelo
    output_path (str): Ruta del archivo de resultados
    model_path (str): Ruta del archivo del modelo
    scaler_path (str): Ruta del archivo del escalador
    """
    
    print("Cargando datos de temperatira")
    data_temp = load_data(input_path)
    e_obj, e_base   =  fijar_energia_anual(ano_to_predict)

    # Cargar modelo
    print("Cargando el modelo a memoria")
    model, scaler   = load_model_and_scaler(model_path, scaler_path)
    ic("Modelo cargado")
    model.summary()
    print("Generando predicciones")
    df_pred = generar_predicciones(
                    data_temp, 
                    e_obj, 
                    e_base, 
                    model_type, 
                    model, 
                    scaler, 
                    output_path)
    
    df_pred_return = df_pred.copy()
    # Muestro una sema especifica del anio con la demanda driver y la proyectada
    # Preprocess de los datos de calendario
    date_range = def_rango_fecha(str(ano_to_predict))

    df_pred.index = date_range


    start_date = f'{ano_to_predict}-06-15'
    end_date = pd.Timestamp(start_date) + pd.DateOffset(days=7)
    week_data = df_pred.loc[start_date:end_date]
    #plot_datos_semana(week_data, start_date, end_date)
    plot_datos_semana(week_data, 
                      start_date, 
                      end_date,
                      ano_to_predict,
                      curva = "",
                      path = "output/predicciones_semana.png")
    return df_pred_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulador de predicciones.')
    parser.add_argument('--ano',    type=int, default= 2030,   
                        help= 'Año a predecir')
    parser.add_argument('--input',  type=str, default= 'datos/temp_sintetico_2.csv', 
                        help='Ruta del archivo de datos de entrada')
    parser.add_argument('--model_type',  type=str, default= 'norm', 
                        help= 'Tipo de modelo')
    parser.add_argument('--output', type=str, default= 'salida/demanda_sim/resultados.csv', 
                        help='Ruta del archivo de resultados')
    parser.add_argument('--model',  type=str, default= 'modelos/UTE_ANN_test', 
                        help='Ruta del archivo del modelo')
    parser.add_argument('--scaler', type=str, default= 'modelos/UTE_ANN_test/scaler_UTE_ANN_test.pkl', 
                        help='Ruta del archivo del escalador')
    args = parser.parse_args()

    try:
        main(args.ano, args.input, args.model_type, args.output, args.model, args.scaler)
    except FileNotFoundError as e:
        print(e)
    except IOError as e:
        print(e)