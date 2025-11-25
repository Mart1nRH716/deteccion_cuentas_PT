from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import *
from model_evaluation import *
from ingestion import *

PATH_SAVE = 'models/'


algorithms_dict = {
		'tree': 'tree_grid_search',
		'random_forest': 'rf_grid_search',
		'logistic_regression': 'logistic_regression_grid_search',
	}
grid_search_dict = {
		'tree_grid_search': {'max_depth': [5,10,15,None], 'min_samples_leaf': [3,5,7]},
		'rf_grid_search': {'n_estimators': [100,300,500,800,1000], 'max_depth': [5,10,15,20,None], 'min_samples_leaf': [3,5,7,11]},
		'logistic_regression_grid_search': {'C': [0.1, 1, 10, 100]}, # Regularization strength (Un valor de 'C' más pequeño indica una regularización más fuerte (penalización más grande), lo que hace que el modelo sea más simple y menos propenso al sobreajuste.)
	}
estimators_dict = {
		'tree': DecisionTreeClassifier(random_state=42),
		'random_forest': RandomForestClassifier(random_state=42),
		'logistic_regression': LogisticRegression(random_state=42, max_iter=500, solver="liblinear"),
	}


def magic_loop(algorithms: list, features: pd.DataFrame, labels: pd.Series) -> list:
	best_estimators = []
	for algorithm in algorithms:
		estimator = estimators_dict[algorithm]
		grid_search_to_look = algorithms_dict[algorithm]
		grid_params = grid_search_dict[grid_search_to_look]
		
		gs = gs = GridSearchCV(estimator,grid_params,scoring='recall',n_jobs=-1,cv=3)
		
		#train
		gs.fit(features, labels)
		#best estimator
		best_estimators.append(gs.best_estimator_)
		
		
	return best_estimators

def save_models(models: list, path: str) -> None:
	for model in models:
		algorithm_name = model.__class__.__name__
		file_path = f"{path}{algorithm_name}.pkl"
		with open(file_path, 'wb') as file:
			pickle.dump(model, file)
		print(f"Model {algorithm_name} saved to {file_path}")


def modeling_pipeline(X_train: pd.DataFrame, y_train: pd.Series, df: pd.DataFrame) -> None:
	best_estimators = {}
	algorithms_dict = ['tree', 'random_forest', 'logistic_regression']


	best_estimators = magic_loop(algorithms_dict, X_train, y_train)
	save_models(best_estimators, PATH_SAVE)




df = load_df("docs/prueba_ml_cuentasMulas.csv")
print("Data frame: ")
print(df)
df = drop_cols(df, ["account_id"])
df = feature_generation(df)
print("Data frame despues deñ  feature generation: ")
print(df)
df = clean_data(df, ["age","transactions_last_24h","total_received_last_7d"])
print("Data frame despues de limpiar datos: ")
print(df)

#Matriz de correlacion:
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("models_reports/correlation_matrix.png")
plt.close()
print("Matriz de correlación generada en models_reports/correlation_matrix.png")

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X = df.drop(["is_mule"], axis=1)
y = df["is_mule"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
modeling_pipeline(X_train, y_train, df)
print("Modelado del pipeline completado.")

models = load_model(PATH_SAVE)
resultados = metric_evaluation(models, X_test, y_test, output_dir='models_reports/')
save_evaluation_results(resultados, output_path='models_reports/evaluation_results.txt')
save_metrics_pkl(resultados, path='models_reports/')