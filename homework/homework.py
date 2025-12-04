# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd  
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import (  
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder  


def cargar_datasets_comprimidos(ruta_carpeta: str) -> list[pd.DataFrame]:
    """Extrae y carga datasets desde archivos ZIP."""
    lista_dfs = []
    archivos_zip = glob(os.path.join(ruta_carpeta, "*"))
    
    for archivo in archivos_zip:
        with zipfile.ZipFile(archivo, "r") as zip_file:
            for contenido in zip_file.namelist():
                with zip_file.open(contenido) as archivo_csv:
                    df = pd.read_csv(archivo_csv, sep=",", index_col=0)
                    lista_dfs.append(df)
    
    return lista_dfs


def limpiar_directorio(directorio: str) -> None:
    """Elimina contenido del directorio y lo recrea."""
    if os.path.exists(directorio):
        archivos = glob(os.path.join(directorio, "*"))
        for archivo in archivos:
            try:
                os.remove(archivo)
            except IsADirectoryError:
                pass
        try:
            os.rmdir(directorio)
        except OSError:
            pass
    
    os.makedirs(directorio, exist_ok=True)


def serializar_modelo_comprimido(ruta_archivo: str, modelo) -> None:
    """Guarda el modelo en formato pickle comprimido con gzip."""
    directorio_padre = os.path.dirname(ruta_archivo)
    limpiar_directorio(directorio_padre)
    
    with gzip.open(ruta_archivo, "wb") as archivo_gz:
        pickle.dump(modelo, archivo_gz)


def limpiar_datos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Limpia y transforma el dataset según las especificaciones."""
    datos = dataframe.copy()
    
    
    datos = datos.rename(columns={"default payment next month": "default"})
    

    datos = datos.loc[datos["MARRIAGE"] != 0]
    datos = datos.loc[datos["EDUCATION"] != 0]
    

    datos["EDUCATION"] = datos["EDUCATION"].apply(
        lambda valor: 4 if valor >= 4 else valor
    )
    
    return datos.dropna()


def dividir_caracteristicas_objetivo(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa el dataset en características (X) y variable objetivo (y)."""
    caracteristicas = dataframe.drop(columns=["default"])
    objetivo = dataframe["default"]
    return caracteristicas, objetivo


def construir_pipeline_optimizacion() -> GridSearchCV:
    """Crea pipeline con OneHotEncoder y RandomForest, y configura GridSearch."""
   
    variables_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    

    codificador = OneHotEncoder(handle_unknown="ignore")
  
    transformador = ColumnTransformer(
        transformers=[("categoricas", codificador, variables_categoricas)],
        remainder="passthrough",
    )
    
    
    bosque_aleatorio = RandomForestClassifier(random_state=42)
    

    pipeline = Pipeline(
        steps=[
            ("preprocesamiento", transformador),
            ("clasificador", bosque_aleatorio),
        ]
    )

    parametros = {
        "clasificador__n_estimators": [100, 200, 500],
        "clasificador__max_depth": [None, 5, 10],
        "clasificador__min_samples_split": [2, 5],
        "clasificador__min_samples_leaf": [1, 2],
    }

    optimizador = GridSearchCV(
        estimator=pipeline,
        param_grid=parametros,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    
    return optimizador


def calcular_metricas_rendimiento(nombre_conjunto: str, valores_reales, valores_predichos) -> dict:
    """Calcula métricas de evaluación del modelo."""
    metricas = {
        "type": "metrics",
        "dataset": nombre_conjunto,
        "precision": precision_score(valores_reales, valores_predichos, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(valores_reales, valores_predichos),
        "recall": recall_score(valores_reales, valores_predichos, zero_division=0),
        "f1_score": f1_score(valores_reales, valores_predichos, zero_division=0),
    }
    return metricas


def generar_matriz_confusion(nombre_conjunto: str, valores_reales, valores_predichos) -> dict:
    """Genera diccionario con la matriz de confusión."""
    matriz = confusion_matrix(valores_reales, valores_predichos)
    
    resultado = {
        "type": "cm_matrix",
        "dataset": nombre_conjunto,
        "true_0": {
            "predicted_0": int(matriz[0][0]),
            "predicted_1": int(matriz[0][1])
        },
        "true_1": {
            "predicted_0": int(matriz[1][0]),
            "predicted_1": int(matriz[1][1])
        },
    }
    
    return resultado


def ejecutar_pipeline_completo() -> None:
    """Función principal que ejecuta todo el flujo de trabajo."""
 
    datasets_crudos = cargar_datasets_comprimidos("files/input")
    datasets_limpios = [limpiar_datos(dataset) for dataset in datasets_crudos]
    

    datos_prueba, datos_entrenamiento = datasets_limpios
    
   
    X_entrenamiento, y_entrenamiento = dividir_caracteristicas_objetivo(datos_entrenamiento)
    X_prueba, y_prueba = dividir_caracteristicas_objetivo(datos_prueba)
    

    modelo_optimizado = construir_pipeline_optimizacion()
    modelo_optimizado.fit(X_entrenamiento, y_entrenamiento)
    

    ruta_modelo = os.path.join("files", "models", "model.pkl.gz")
    serializar_modelo_comprimido(ruta_modelo, modelo_optimizado)
    
 
    predicciones_prueba = modelo_optimizado.predict(X_prueba)
    predicciones_entrenamiento = modelo_optimizado.predict(X_entrenamiento)
  
    metricas_entrenamiento = calcular_metricas_rendimiento(
        "train", y_entrenamiento, predicciones_entrenamiento
    )
    metricas_prueba = calcular_metricas_rendimiento(
        "test", y_prueba, predicciones_prueba
    )
    
  
    confusion_entrenamiento = generar_matriz_confusion(
        "train", y_entrenamiento, predicciones_entrenamiento
    )
    confusion_prueba = generar_matriz_confusion(
        "test", y_prueba, predicciones_prueba
    )
    
  
    Path("files/output").mkdir(parents=True, exist_ok=True)
    
    ruta_metricas = "files/output/metrics.json"
    with open(ruta_metricas, "w", encoding="utf-8") as archivo_metricas:
        archivo_metricas.write(json.dumps(metricas_entrenamiento) + "\n")
        archivo_metricas.write(json.dumps(metricas_prueba) + "\n")
        archivo_metricas.write(json.dumps(confusion_entrenamiento) + "\n")
        archivo_metricas.write(json.dumps(confusion_prueba) + "\n")


if __name__ == "__main__":
    ejecutar_pipeline_completo()