# Prueba ML Fraude

Este proyecto es un pipeline de **Machine Learning** para detección de cuentas “mula” en transacciones financieras.  
El proyecto incluye **ingestión de datos**, **generación de features**, **entrenamiento de modelos**, **evaluación de modelos**, y **reportes**.  



## Instalación y ejecución

### Requisitos

- Docker ≥ 20.10
- docker-compose 
- Python 3.11 (solo en caso de no utilzar docker)

---

###  Ejecutar con Docker

Desde la raíz del proyecto:

```bash
sudo docker compose up --build
```
Si no se tienen los archivos en las carpetas models y models_reports, el contenedor creará y pasará los archivos a dichas carpetas que se encuntran dentro del proyecto.

En caso de no ejecutar el proyecto con docker, se tiene que crear el entorno virtual e instalar las dependecias.
```bash
python3 -m venv env

venv\Scripts\activate     # En Windows
source env/bin/activate # En Caso de Linux

pip install -r requirements.txt

python modeling.py
```


