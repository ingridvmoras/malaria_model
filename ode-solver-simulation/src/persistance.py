import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_evenness_before_dry(y_out, p):
    """
    Extrae los valores de evenness antes de cada temporada seca.
    
    Argumentos:
        y_out : objeto devuelto por `odeSolver` con las soluciones.
        p     : diccionario con parámetros del modelo.
    
    Salida:
        df_evenness : DataFrame con los valores de Evenness antes de la temporada seca en cada año.
    """
    # Lista para almacenar valores de evenness antes de cada temporada seca
    evenness_values = []
    years = int(p['t_f'] / p['year_duration'])  # Número de años simulados
    
    for year in range(years):
        t_dry_start = year * p['year_duration'] + p['t_dry']  # Tiempo de inicio de temporada seca
        t_index = np.argmin(np.abs(y_out.t - t_dry_start))  # Encontrar el índice más cercano

        if t_index < len(y_out.t):
            evenness_values.append((y_out.t[t_index], y_out.persister_y[t_index]))

    # Convertimos la lista en un DataFrame
    df_evenness = pd.DataFrame(evenness_values, columns=["Time", "Evenness"])
    
    return df_evenness