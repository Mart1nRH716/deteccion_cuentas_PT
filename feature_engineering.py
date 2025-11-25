import pandas as pd


"""
	Función que genera nuevas características en el DataFrame.
	Recibe:
		df (DataFrame): El DataFrame original.
	Regresa:
		DataFrame: El DataFrame con nuevas características generadas. Para este caso, se pide:
	ratio = total_received_last_7d / (transactions_last_24h + 1)
	high_activity_flag = 1 si transactions_last_24h > 20
"""
def feature_generation(df: pd.DataFrame) -> pd.DataFrame:
	df['ratio'] = df['total_received_last_7d'] / (df['transactions_last_24h'] + 1)
	df['high_activity_flag'] = (df['transactions_last_24h'] > 20).astype(int)
	
	return df