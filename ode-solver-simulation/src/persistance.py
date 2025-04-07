import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_evenness_before_dry(df, p):
    """
    Extrae los valores de evenness antes de cada temporada seca.

    Argumentos:
        df : DataFrame devuelto por `odeSolver` con las soluciones.
        p  : diccionario con parámetros del modelo.

    Salida:
        df_evenness : DataFrame con los valores de Evenness antes de la temporada seca en cada año.
    """
    # Lista para almacenar valores de evenness antes de cada temporada seca
    evenness_values = []
    years = int(p['t_f'] / p['year_duration'])  # Número de años simulados

    for year in range(years):
        t_dry_start = year * p['year_duration'] + p['t_dry']  # Tiempo de inicio de temporada seca
        t_index = np.argmin(np.abs(df['Time'] - t_dry_start))  # Encontrar el índice más cercano

        if t_index < len(df):
            evenness_values.append((df['Time'].iloc[t_index], df['Evenness'].iloc[t_index]))

    # Convertimos la lista en un DataFrame
    df_evenness = pd.DataFrame(evenness_values, columns=["Time", "Evenness"])

    return df_evenness

# Dividir el conjunto de datos excluyendo los tiempos de las temporadas secas
def split_dataset_excluding_dry_season(df, p):
    """
    Divide el conjunto de datos excluyendo los tiempos de las temporadas secas.

    Argumentos:
        df : DataFrame con los datos.
        p  : diccionario con parámetros del modelo.

    Salida:
        df_nomay : DataFrame con los datos excluyendo los tiempos de las temporadas secas.
        df_dry : DataFrame con los datos de los tiempos de las temporadas secas.
    """
    # Lista para almacenar los índices de los tiempos de las temporadas secas
    dry_season_indices = []
    years = int(p['t_f'] / p['year_duration'])  # Número de años simulados

    for year in range(years):
        t_dry_start = year * p['year_duration'] + p['t_dry']  # Tiempo de inicio de temporada seca
        t_index = np.argmin(np.abs(df['Time'] - t_dry_start))  # Encontrar el índice más cercano

        if t_index < len(df):
            dry_season_indices.append(t_index)

    # Crear un DataFrame excluyendo los tiempos de las temporadas secas
    df_nomay = df.drop(dry_season_indices).copy()
    # Crear un DataFrame con solo los tiempos de las temporadas secas
    df_dry = df.iloc[dry_season_indices].copy()

    return df_nomay, df_dry

# Ejemplo de uso
# df_nomay, df_dry = split_dataset_excluding_dry_season(kid_t, p)