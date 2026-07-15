from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de Inteligencia Artificial de Gimnasio Vórtice S.A.C. activo y operando",
        "estado": "Online",
        "api_docs": "/docs"
    }

# Cargar el modelo y el escalador cuando arranca el servidor
modelo = joblib.load('modelo_random_forest_vortice.joblib')
scaler = joblib.load('escalador_vortice.joblib')

# Definir la estructura de datos que va a recibir la API
class DatosCliente(BaseModel):
    asistencia_semanal: float
    antiguedad_meses: float
    consumo_barra: float
    uso_app: int
    genero_masculino: int
    membresia_trimestral: int
    membresia_anual: int

@app.post("/predecir_fuga")
def predecir(cliente: DatosCliente):
    # Convertir los datos recibidos a un formato que el modelo entienda
    datos = np.array([[
        cliente.asistencia_semanal,
        cliente.antiguedad_meses,
        cliente.consumo_barra,
        cliente.uso_app,
        cliente.genero_masculino,
        cliente.membresia_trimestral,
        cliente.membresia_anual
    ]])
    
    # Escalar los datos con el scaler cargado
    datos_escalados = scaler.transform(datos)
    
    # Hacer la predicción de probabilidad
    probabilidad = modelo.predict_proba(datos_escalados)[0][1] # Probabilidad de fuga
    alerta = bool(probabilidad > 0.70) # Alerta si supera el 70%
    
    return {
        "probabilidad_desercion": round(float(probabilidad), 4),
        "alerta_de_fuga": alerta
    }
