from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1. Cargar el modelo y el scaler desde los archivos .pkl
# -------------------------------------------------------------------
# with open('modelo_regresion_logistica.pkl', 'rb') as archivo_modelo:
with open('modelo_bosque_aleatorio.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# -------------------------------------------------------------------
# 2. Mapa de índices -> texto de clase
#    Ajusta estos nombres según las clases reales de tu problema.
# -------------------------------------------------------------------
class_label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

# -------------------------------------------------------------------
# 3. Definir las columnas en el mismo orden que se usaron en entrenamiento
# -------------------------------------------------------------------
columnas = [
    'Gender',  # str o num, según tu preprocesado
    'Age',
    'Height',
    'Weight',
    'family_history_with_overweight',
    'FAVC',
    'FCVC',
    'NCP',
    'CAEC',
    'SMOKE',
    'CH2O',
    'SCC',
    'FAF',
    'TUE',
    'CALC',
    'MTRANS'
]

# -------------------------------------------------------------------
# 4. Crear la aplicación FastAPI
# -------------------------------------------------------------------
app = FastAPI(title="Predicción de Obesidad")

# -------------------------------------------------------------------
# 5. Modelo de entrada (pydantic)
#    Ajusta los tipos (str, float, etc.) a como esperas los datos.
# -------------------------------------------------------------------
class ObesityInput(BaseModel):
    Gender: str
    Age: float
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

# -------------------------------------------------------------------
# 6. Endpoint de predicción
# -------------------------------------------------------------------
@app.post("/prediccion/")
async def predecir_obesidad(entrada: ObesityInput):
    """
    Recibe las características de una persona (dataset Obesidad) y devuelve
    la clase de obesidad predicha (en texto) y las probabilidades por cada clase.
    """
    try:
        # a) Convertir la entrada en DataFrame
        datos_entrada = pd.DataFrame([entrada.dict()], columns=columnas)

        # b) Escalar las características
        datos_entrada_scaled = scaler.transform(datos_entrada)

        # c) Predicción (devuelve índice de clase, ej. 0..6)
        prediccion = modelo.predict(datos_entrada_scaled)
        clase_idx = int(prediccion[0])  # Convertir a entero

        # d) Obtener probabilidades por clase (opcional)
        probabilidades = modelo.predict_proba(datos_entrada_scaled)[0]

        # e) Mapear clase índice -> texto
        clase_texto = class_label_map.get(clase_idx, f"Desconocido_{clase_idx}")

        # f) Construir el dict de probabilidades con texto de clase
        #    Siguiendo el orden de 'modelo.classes_' (ej.: [0, 1, 2, 3, 4, 5, 6])
        #    Si 'modelo.classes_' es [0,1,2,3,4,5,6], podemos usar un bucle.
        resultado_prob = {}
        for i, idx_clase in enumerate(modelo.classes_):
            # idx_clase = 0..6, i = 0..6
            nombre_clase = class_label_map.get(idx_clase, f"Desconocido_{idx_clase}")
            resultado_prob[nombre_clase] = float(probabilidades[i])

        # g) Crear la respuesta final
        resultado = {
            "ClasePredicha": clase_texto,
            "Probabilidades": resultado_prob
        }

        return resultado

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
