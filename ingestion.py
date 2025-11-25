import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


'''
Función que carga un objeto de tipo DataFrame desde un archivo CSV.
Recibe:
	filename (str): Ruta del archivo CSV a cargar.
Regresa:
	DataFrame: Un DataFrame de pandas cargado desde el archivo CSV.
'''	
def load_df(filename: str) -> pd.DataFrame:
	return pd.read_csv(filename)


'''Función que guarda un objeto de tipo DataFrame en un archivo CSV.
Recibe:
	df (DataFrame): El DataFrame a guardar.
	filename (str): Ruta del archivo CSV donde se guardará el DataFrame.
Regresa:
	None
'''
def save_df(df: pd.DataFrame, filename: str) -> None:
	df.to_csv(filename, index=False)

	

'''
Función que ingesta un archivo CSV y devuelve un DataFrame con los títulos de las columnas normalizadas.
Recibe:
	file_path (str): Ruta del archivo CSV a ingestar.
Regresa:
	DataFrame: Un DataFrame de pandas cargado desde el archivo CSV.
'''
def ingest_file(file_path: str) -> pd.DataFrame:
	df = load_df(file_path)
	df.columns = df.columns.str.strip().str.lower()
	return df


''''
Función que guarda un DataFrame en un archivo CSV después de la ingesta.
Recibe:
	df (DataFrame): El DataFrame a guardar.
	output_path (str): Ruta del archivo CSV donde se guardará el DataFrame.
Regresa:
	None
'''
def save_ingested_data(df: pd.DataFrame, output_path: str) -> None:
	save_df(df, output_path)



"""
Función que elimina columnas específicas de un DataFrame.
Recibe:
	df (DataFrame): El DataFrame del cual se eliminarán las columnas.
	cols (list): Lista de nombres de columnas a eliminar.
Regresa:
	DataFrame: El DataFrame sin las columnas especificadas.
"""
def drop_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
	return df.drop(columns=cols, errors='ignore') if cols else df


"""
	Función que limpia un DataFrame eliminando filas con valores nulos en columnas específicas, elimina valores atípicos y 
	normaliza.
"""

def clean_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
	# Eliminar filas con valores nulos en las columnas especificadas
	df = df.dropna(subset=cols)
	
	cuartil_1 = df[cols].quantile(0.25)
	cuartil_3 = df[cols].quantile(0.75)
	rango_intercuartilico = cuartil_3 - cuartil_1
	limite_inferior = cuartil_1 - 1.5 * rango_intercuartilico
	limite_superior = cuartil_3 + 1.5 * rango_intercuartilico
	for col in cols:
		df = df[(df[col] >= limite_inferior[col]) & (df[col] <= limite_superior[col])]
	# Normalizar las columnas especificadas
	escaler = StandardScaler()
	df[cols] = escaler.fit_transform(df[cols])
	
	return df