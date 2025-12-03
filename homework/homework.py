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
    # Cargar y limpiar datos
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