import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns


"""
    Funcion para cargar múltiples modelos desde una ruta especificada.
    Devuelve un diccionario con los modelos cargados.
"""
def load_model(models_paths: str) -> dict:
	#Buscamos todos los archivos .pkl en la ruta especificada
	models = {}
	for filename in os.listdir(models_paths):
		if filename.endswith('.pkl'):
			model_name = filename[:-4]
			models[model_name] = load_single_model(os.path.join(models_paths, filename))
	return models

"""
    Carga un modelo desde un archivo pickle.
"""
def load_single_model(file_path: str) -> object:
	with open(file_path, 'rb') as file:
		model = pickle.load(file)
	return model

"""
    Evalúa múltiples modelos y guarda los resultados y gráficas en la carpeta especificada.
"""
def metric_evaluation(models: dict, X_test, y_test, output_dir: str):
	os.makedirs(output_dir, exist_ok=True) 
	results = {}

	for model_name, model in models.items():
		print("="*60)
		print(f"Model: {model_name}")

		# Predicciones
		y_pred = model.predict(X_test)

		# Probabilidades
		y_proba = None
		if hasattr(model, "predict_proba"):
			y_proba = model.predict_proba(X_test)[:, 1]

		# Métricas
		acc = accuracy_score(y_test, y_pred)
		prec = precision_score(y_test, y_pred, average="weighted")
		rec = recall_score(y_test, y_pred, average="weighted")
		f1 = f1_score(y_test, y_pred, average="weighted")

		results[model_name] = {
			"accuracy": acc,
			"precision": prec,
			"recall": rec,
			"f1": f1
		}

		# Guardar reporte en txt
		report_path = os.path.join(output_dir, f"{model_name}_report.txt")
		with open(report_path, "w") as f:
			f.write(f"Model: {model_name}\n")
			f.write(f"Accuracy: {acc:.4f}\n")
			f.write(f"Precision: {prec:.4f}\n")
			f.write(f"Recall: {rec:.4f}\n")
			f.write(f"F1-score: {f1:.4f}\n\n")
			f.write("Classification Report:\n")
			f.write(classification_report(y_test, y_pred))

			try:
				params = model.best_params_
				f.write("Mejores Hiperparámetros:\n")
				for param, value in params.items():
					f.write(f"  {param}: {value}\n")
			except AttributeError:
				f.write("Mejores Hiperparámetros: No disponible (modelo no proveniente de GridSearchCV)\n")

				f.write("\nClassification Report:\n")
				f.write(classification_report(y_test, y_pred))

		# Matriz de confusión
		cm = confusion_matrix(y_test, y_pred)
		plt.figure(figsize=(6,4))
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
		plt.title(f"Confusion Matrix - {model_name}")
		plt.xlabel("Predicted")
		plt.ylabel("True")
		cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
		plt.savefig(cm_path)
		plt.close()

		# Curva ROC
		if y_proba is not None:
			auc = roc_auc_score(y_test, y_proba)
			fpr, tpr, _ = roc_curve(y_test, y_proba)

			plt.figure(figsize=(6,4))
			plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
			plt.plot([0,1], [0,1], linestyle="--", color="gray")
			plt.xlabel("False Positive Rate")
			plt.ylabel("True Positive Rate")
			plt.title(f"ROC Curve - {model_name}")
			plt.legend()
			roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
			plt.savefig(roc_path)
			plt.close()

		print(f"Resultados y gráficas guardados para {model_name}")

	return results

def save_evaluation_results(results: dict, output_path: str) -> None:
	with open(output_path, 'w') as file:
		for model_name, score in results.items():
			file.write(f"Model: {model_name}, Score: {score}\n")
	print(f"Resultados guardados en: {output_path}")

"""
    Guarda un DataFrame con métricas en formato pickle.
    El archivo se llamará 'metricas_offline.pkl' y se guardará en la carpeta indicada.
"""
def save_metrics_pkl(resultados: dict, path: str) -> None:
    os.makedirs(path, exist_ok=True)  # Crea la carpeta si no existe
    file_path = os.path.join(path, "metricas_offline.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(resultados, f)
    print(f"Métricas guardadas en: {file_path}")