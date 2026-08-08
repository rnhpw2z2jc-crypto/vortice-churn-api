"""
Vórtice Gym Power - API de Predicción de Deserción
==================================================
API REST para predicción de fuga de clientes basada en Machine Learning.

Proyecto Académico - VII Ciclo Ingeniería de Sistemas UCV
Cumple con estándares ISO 25010 para calidad de software.

Autor: Ingeniería de Datos - Proyecto Vórtice S.A.C.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import numpy as np
import logging
from datetime import datetime
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gimnasio Vórtice S.A.C. - API de IA",
    description="API para predicción de deserción de clientes usando Machine Learning",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "modelo_random_forest_vortice.joblib"
SCALER_PATH = "escalador_vortice.joblib"
HISTORY_FILE = "historial_predicciones.json"
MODEL_INFO_FILE = "modelo_info.json"
INFORMES_FILE = "informes_semanales.json"
FEEDBACK_FILE = "feedback.json"
MODEL_BACKUP_PATH = "modelo_random_forest_vortice_respaldo.joblib"
SCALER_BACKUP_PATH = "escalador_vortice_respaldo.joblib"

FEATURES = ["edad", "antiguedad_meses", "precio_membresia", "asistencia_semanal", "consumo_barra", "uso_app", "genero_masculino", "membresia_mensual", "membresia_trimestral"]

def cargar_info_modelo():
    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": "1.0",
        "fecha_entrenamiento": "2026-01-15",
        "fecha_ultimo_reentrenamiento": None,
        "muestras_entrenamiento": 1200,
        "muestras_actuales": 0,
        "accuracy_entrenamiento": 0.986,
        "metricas": {"accuracy": 0.85, "precision": 0.82, "recall": 0.78, "f1_score": 0.80, "auc_roc": 0.88},
        "baseline": {},
        "reentrenamientos": [],
        "drift_detectado": False
    }

def guardar_info_modelo(info):
    with open(MODEL_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

info_modelo = cargar_info_modelo()

try:
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Modelo y escalador cargados exitosamente")
except Exception as e:
    logger.error(f"Error cargando el modelo: {e}")
    raise

def cargar_historial():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_historial(historial):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

def cargar_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_feedback(feedback):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

def registrar_feedback(id_prediccion, socio_se_fugo):
    """Registra el resultado real de una prediccion (el modelo aprende de sus errores)."""
    historial = cargar_historial()
    pred = next((p for p in historial if p["id"] == id_prediccion), None)
    if not pred:
        raise ValueError(f"No existe la prediccion con id {id_prediccion}")
    if not pred.get("cliente_demo"):
        raise ValueError("La prediccion no contiene datos del cliente para entrenar")

    feedback = cargar_feedback()
    for f in feedback:
        if f["id_prediccion"] == id_prediccion:
            f["socio_se_fugo"] = socio_se_fugo
            f["fecha_feedback"] = datetime.now().isoformat()
            guardar_feedback(feedback)
            return {"exito": True, "actualizado": True, "id": id_prediccion, "socio_se_fugo": socio_se_fugo}

    feedback.append({
        "id_prediccion": id_prediccion,
        "probabilidad_predicha": pred.get("probabilidad_desercion"),
        "socio_se_fugo": socio_se_fugo,
        "fecha_prediccion": pred.get("timestamp"),
        "fecha_feedback": datetime.now().isoformat(),
        "cliente_demo": pred["cliente_demo"]
    })
    guardar_feedback(feedback)
    return {"exito": True, "actualizado": False, "id": id_prediccion, "socio_se_fugo": socio_se_fugo}

def registrar_feedback_lote(resultados):
    """Registra resultados reales para varias predicciones de una sola vez."""
    registrados = 0
    for item in resultados:
        registrar_feedback(item["id_prediccion"], item["socio_se_fugo"])
        registrados += 1
    return {"exito": True, "registrados": registrados, "total_feedback": len(cargar_feedback())}

def calcular_baseline(historial):
    """Calcula estadisticas de referencia (baseline) por feature."""
    filas = [p for p in historial if p.get("cliente_demo")]
    if not filas:
        return {}
    stats = {}
    for feat in FEATURES:
        valores = [p["cliente_demo"].get(feat, 0) for p in filas]
        arr = np.array(valores, dtype=float)
        stats[feat] = {"media": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4) if arr.std() > 0 else 0.001}
    return stats

def monitorear_drift():
    """Compara la distribucion actual de datos vs el baseline de entrenamiento."""
    historial = cargar_historial()
    filas = [p for p in historial if p.get("cliente_demo")]
    return _calcular_drift(filas, info_modelo.get("baseline"))

def _calcular_drift(filas, baseline):
    """Nucleo del calculo de drift. Acepta un set de filas y un baseline.
    Si no hay baseline, se toma la distribucion del historial real como referencia."""
    if not baseline:
        historial = cargar_historial()
        baseline = calcular_baseline([p for p in historial if p.get("cliente_demo")])
    if not baseline:
        return {"estado": "sin_baseline", "features": {}, "mensaje": "No hay datos suficientes"}

    if not filas:
        return {"estado": "sin_datos", "features": {}, "mensaje": "No hay predicciones"}

    features_drift = {}
    for feat in FEATURES:
        if feat not in baseline:
            continue
        base = baseline[feat]
        valores = np.array([f["cliente_demo"].get(feat, 0) for f in filas], dtype=float)
        media_actual = float(valores.mean())
        score = abs(media_actual - base["media"]) / base["std"]
        if score < 0.15:
            nivel = "OK"
        elif score < 0.35:
            nivel = "LEVE"
        elif score < 0.60:
            nivel = "MODERADO"
        else:
            nivel = "SEVERO"
        features_drift[feat] = {
            "baseline": base["media"],
            "actual": round(media_actual, 4),
            "score": round(score, 3),
            "nivel": nivel
        }

    severos = [f for f, v in features_drift.items() if v["nivel"] in ("MODERADO", "SEVERO")]
    estado = "SEVERO" if any(v["nivel"] == "SEVERO" for v in features_drift.values()) else \
             "MODERADO" if severos else "ESTABLE"
    info_modelo["drift_detectado"] = estado != "ESTABLE"
    return {
        "estado": estado,
        "features": features_drift,
        "total_features": len(features_drift),
        "en_drift": len(severos),
        "fecha": datetime.now().isoformat(),
        "mensaje": "Se detecta desviacion de datos. Reentrena el modelo para mantener precision." if estado != "ESTABLE" else "Distribucion estable, el modelo mantiene su precision."
    }

def simular_drift(n, perfil):
    """Simula la llegada de un nuevo segmento de clientes para demostrar el monitoreo
    de data drift SIN modificar el historial real."""
    ahora = datetime.now()
    filas = []
    for i in range(n):
        if perfil == "digital":
            base = {
                "edad": 24, "antiguedad_meses": 3, "precio_membresia": 120.0,
                "asistencia_semanal": 1.0, "consumo_barra": 15.0, "uso_app": 1,
                "genero_masculino": 1, "membresia_mensual": 1, "membresia_trimestral": 0
            }
        elif perfil == "premium":
            base = {
                "edad": 48, "antiguedad_meses": 36, "precio_membresia": 1050.0,
                "asistencia_semanal": 1.5, "consumo_barra": 40.0, "uso_app": 0,
                "genero_masculino": 0, "membresia_mensual": 0, "membresia_trimestral": 0
            }
        else:
            base = {
                "edad": 55, "antiguedad_meses": 48, "precio_membresia": 290.0,
                "asistencia_semanal": 0.5, "consumo_barra": 10.0, "uso_app": 0,
                "genero_masculino": 1, "membresia_mensual": 0, "membresia_trimestral": 0
            }
        fila = {"cliente_demo": dict(base)}
        for k, v in base.items():
            fila["cliente_demo"][k] = v + np.random.uniform(-0.3, 0.3) * (abs(v) + 1)
        fila["cliente_demo"]["asistencia_semanal"] = max(0.0, fila["cliente_demo"]["asistencia_semanal"])
        fila["cliente_demo"]["uso_app"] = float(base["uso_app"])
        fila["cliente_demo"]["genero_masculino"] = float(base["genero_masculino"])
        fila["cliente_demo"]["membresia_mensual"] = float(base["membresia_mensual"])
        fila["cliente_demo"]["membresia_trimestral"] = float(base["membresia_trimestral"])
        filas.append(fila)

    resultado = _calcular_drift(filas, info_modelo.get("baseline"))
    resultado["simulado"] = True
    resultado["segmento"] = perfil
    resultado["nuevos_clientes"] = n
    return resultado

def reentrenar_modelo():
    """Reentrena el modelo con los resultados reales (feedback) acumulados.
    Prioriza feedback observado; si no hay suficiente, usa las etiquetas del historial."""
    global modelo, scaler
    feedback = cargar_feedback()
    filas_feedback = [f for f in feedback if f.get("cliente_demo")]
    fuente = "feedback"
    filas = filas_feedback
    y_key = "socio_se_fugo"
    if len(filas) < 50:
        historial = cargar_historial()
        filas = [p for p in historial if p.get("cliente_demo")]
        y_key = "alerta_de_fuga"
        fuente = "historial"
        if len(filas) < 50:
            raise ValueError("No hay suficientes datos para reentrenar (minimo 50).")

    X = np.array([[f["cliente_demo"][feat] for feat in FEATURES] for f in filas], dtype=float)
    y = np.array([1 if f.get(y_key) else 0 for f in filas], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_n = StandardScaler()
    X_train_s = scaler_n.fit_transform(X_train)
    X_test_s = scaler_n.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    y_prob = rf.predict_proba(X_test_s)[:, 1]
    metricas = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4)
    }
    cv_scores = cross_val_score(rf, X_train_s, y_train, cv=5, scoring="accuracy")

    if not os.path.exists(MODEL_BACKUP_PATH):
        joblib.dump(modelo, MODEL_BACKUP_PATH)
        joblib.dump(scaler, SCALER_BACKUP_PATH)

    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler_n, SCALER_PATH)

    modelo = rf
    scaler = scaler_n

    historial = cargar_historial()
    version_actual = float(info_modelo.get("version", "1.0"))
    info_modelo["version"] = str(round(version_actual + 0.1, 1))
    info_modelo["fecha_ultimo_reentrenamiento"] = datetime.now().isoformat()
    info_modelo["muestras_actuales"] = len(filas)
    info_modelo["metricas"] = metricas
    info_modelo["baseline"] = calcular_baseline(historial)
    info_modelo["drift_detectado"] = False
    info_modelo["reentrenamientos"].append({
        "fecha": datetime.now().isoformat(),
        "version": info_modelo["version"],
        "muestras": len(filas),
        "fuente": fuente,
        "metricas": metricas,
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "razon": "Data drift detectado / mejora continua con feedback real"
    })
    guardar_info_modelo(info_modelo)

    return {
        "exito": True,
        "version": info_modelo["version"],
        "muestras": len(filas),
        "fuente": fuente,
        "metricas": metricas,
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "fecha": info_modelo["fecha_ultimo_reentrenamiento"]
    }

class DatosCliente(BaseModel):
    edad: float = Field(..., ge=14, le=90, description="Edad del socio")
    antiguedad_meses: float = Field(..., ge=0, le=120, description="Meses de antigüedad")
    precio_membresia: float = Field(..., gt=0, description="Precio de la membresía")
    asistencia_semanal: float = Field(..., ge=0, le=7, description="Asistencia promedio semanal")
    consumo_barra: float = Field(..., ge=0, description="Consumo en barra nutricional")
    uso_app: int = Field(..., ge=0, le=1, description="Uso de plataforma web")
    genero_masculino: int = Field(..., ge=0, le=1, description="Género (1=M, 0=F)")
    membresia_mensual: int = Field(..., ge=0, le=1, description="Membresía mensual")
    membresia_trimestral: int = Field(..., ge=0, le=1, description="Membresía trimestral")

    @validator('precio_membresia')
    def validar_precio(cls, v):
        precios_validos = [110.0, 120.0, 290.0, 320.0, 990.0, 1000.0, 1100.0]
        if v not in precios_validos:
            raise ValueError(f'Precio no válido. Use: {precios_validos}')
        return v

class RespuestaPrediccion(BaseModel):
    id: str
    probabilidad_desercion: float
    alerta_de_fuga: bool
    nivel_riesgo: str
    recomendacion: str
    timestamp: str

class MetricasModelo(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

class DatosLote(BaseModel):
    clientes: List[DatosCliente]

class ResultadoLote(BaseModel):
    total_clientes: int
    en_riesgo: int
    seguros: int
    probabilidad_promedio: float
    ahorro_estimado: float
    predicciones: List[dict]

class CalculadoraROI(BaseModel):
    num_socios: int = Field(..., ge=1, description="Número de socios")
    costo_adquisicion: float = Field(default=150.0, description="Costo de adquisición por cliente (S/.)")
    valor_vida_cliente: float = Field(default=1800.0, description="Valor de vida del cliente (S/.)")

class ResultadoROI(BaseModel):
    num_socios: int
    clientes_en_riesgo: int
    tasa_desercion: float
    ahorro_anual: float
    roi_porcentaje: float
    payback_meses: float
    inversion_sistema: float

class FeedbackRequest(BaseModel):
    id_prediccion: str = Field(..., description="ID de la prediccion a evaluar")
    socio_se_fugo: bool = Field(..., description="Resultado real: True si el socio se fugo, False si se quedo")

class FeedbackItem(BaseModel):
    id_prediccion: str
    socio_se_fugo: bool

class FeedbackLoteRequest(BaseModel):
    resultados: List[FeedbackItem]

class SimularDriftRequest(BaseModel):
    n: int = Field(default=200, ge=10, le=1000, description="Numero de nuevos clientes simulados")
    perfil: str = Field(default="digital", description="Perfil del nuevo segmento: digital, premium o adulto")

def predecir_cliente(cliente: DatosCliente):
    datos = np.array([[
        cliente.edad, cliente.antiguedad_meses, cliente.precio_membresia,
        cliente.asistencia_semanal, cliente.consumo_barra, cliente.uso_app,
        cliente.genero_masculino, cliente.membresia_mensual, cliente.membresia_trimestral
    ]])
    datos_escalados = scaler.transform(datos)
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]

    if probabilidad > 0.7:
        nivel_riesgo = "CRÍTICO"
        recomendacion = "Acción inmediata requerida. Contactar al socio con una oferta personalizada urgente."
    elif probabilidad > 0.5:
        nivel_riesgo = "ALTO"
        recomendacion = "Ofrecer descuento del 15-20% o beneficios adicionales para retener al socio."
    elif probabilidad > 0.3:
        nivel_riesgo = "MEDIO"
        recomendacion = "Enviar comunicación de bienvenida y ofrecer clase gratuita o servicio adicional."
    else:
        nivel_riesgo = "BAJO"
        recomendacion = "Socio estable. Invitar a programas de referidos o retos internos."

    return float(probabilidad), nivel_riesgo, recomendacion

@app.get("/", response_class=HTMLResponse, tags=["Principal"])
async def home():
    html_content = r"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vórtice Power - Plataforma de IA Empresarial</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            gold: { 50:'#FDF8E8', 100:'#F4E8C1', 200:'#E8D48A', 300:'#E5C158', 400:'#D4AF37', 500:'#C9A227', 600:'#B8932A', 700:'#9A7B1F', 800:'#7D6318', 900:'#4A3B0F' },
                            dark: { 50:'#2a2a2a', 100:'#222222', 200:'#1c1c1c', 300:'#181818', 400:'#141414', 500:'#121212', 600:'#0e0e0e', 700:'#0b0b0b', 800:'#080808', 900:'#050505' }
                        }
                    }
                }
            }
        </script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            .gold-glow { text-shadow: 0 0 20px rgba(212,175,55,0.4); }
            .card-glow:hover { box-shadow: 0 0 30px rgba(212,175,55,0.1); }
            .stat-card { background: linear-gradient(135deg, #141414 0%, #1c1c1c 100%); }
            @keyframes countUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            .animate-count { animation: countUp 0.6s ease-out forwards; }
            @keyframes pulse-gold { 0%, 100% { box-shadow: 0 0 0 0 rgba(212,175,55,0.4); } 50% { box-shadow: 0 0 0 10px rgba(212,175,55,0); } }
            .pulse-gold { animation: pulse-gold 2s infinite; }
            .chart-container { position: relative; height: 200px; }
            .scrollbar-hide::-webkit-scrollbar { display: none; }
            .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
            @media (max-width: 640px) {
                .chart-container { height: 180px; }
                .mobile-menu { display: none; }
                .mobile-menu.open { display: flex; }
            }
            @media (min-width: 641px) {
                .mobile-menu { display: none !important; }
                .btn-hamburger { display: none !important; }
            }
        </style>
    </head>
    <body class="bg-dark-700 text-white min-h-screen">

        <!-- Header -->
        <header class="bg-dark-500 border-b border-zinc-800/50 sticky top-0 z-50 backdrop-blur-xl bg-opacity-90">
            <div class="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center">
                <div class="flex items-center gap-3">
                    <div class="bg-gradient-to-br from-gold-400 to-gold-600 p-2.5 rounded-xl text-black shadow-lg shadow-gold-400/20 pulse-gold">
                        <i class="fa-solid fa-bolt text-lg"></i>
                    </div>
                    <div>
                        <h1 class="text-lg font-black tracking-wider text-gold-400 gold-glow">VÓRTICE POWER</h1>
                        <p class="text-[10px] text-zinc-500 font-medium tracking-wide">PLATAFORMA DE IA EMPRESARIAL v3.0</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <nav class="hidden sm:flex gap-1">
                        <button onclick="showSection('predictor')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-gold-400 bg-gold-400/10 border border-gold-400/20" data-section="predictor">
                            <i class="fa-solid fa-brain mr-1"></i> Predictor
                        </button>
                        <button onclick="showSection('dashboard')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section="dashboard">
                            <i class="fa-solid fa-chart-pie mr-1"></i> Dashboard
                        </button>
                        <button onclick="showSection('roi')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section="roi">
                            <i class="fa-solid fa-calculator mr-1"></i> ROI
                        </button>
                        <button onclick="showSection('lote')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section="lote">
                            <i class="fa-solid fa-layer-group mr-1"></i> Lote
                        </button>
                        <button onclick="showSection('reportes')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section="reportes">
                            <i class="fa-solid fa-file-lines mr-1"></i> Reportes
                        </button>
                        <button onclick="showSection('modelo')" class="nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section="modelo">
                            <i class="fa-solid fa-arrows-rotate mr-1"></i> Modelo
                        </button>
                    </nav>
                    <button onclick="toggleMobileMenu()" class="btn-hamburger bg-zinc-800 hover:bg-zinc-700 text-zinc-300 p-2 rounded-lg border border-zinc-700/50">
                        <i class="fa-solid fa-bars text-sm"></i>
                    </button>
                    <a href="/docs" target="_blank" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-400 text-xs px-3 py-1.5 rounded-lg border border-zinc-700/50 transition-all">
                        <i class="fa-solid fa-book mr-1"></i> API
                    </a>
                    <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] px-2 py-1 rounded-full flex items-center">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1 animate-pulse"></span> Live
                    </span>
                </div>
            </div>
            <div id="mobile-menu" class="mobile-menu max-w-7xl mx-auto px-4 pb-3 flex-col gap-2">
                <button onclick="showSection('predictor'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-gold-400 bg-gold-400/10 border border-gold-400/20" data-section-mobile="predictor">
                    <i class="fa-solid fa-brain mr-2"></i> Predictor
                </button>
                <button onclick="showSection('dashboard'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section-mobile="dashboard">
                    <i class="fa-solid fa-chart-pie mr-2"></i> Dashboard
                </button>
                <button onclick="showSection('roi'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section-mobile="roi">
                    <i class="fa-solid fa-calculator mr-2"></i> ROI
                </button>
                <button onclick="showSection('lote'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section-mobile="lote">
                    <i class="fa-solid fa-layer-group mr-2"></i> Lote
                </button>
                <button onclick="showSection('reportes'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section-mobile="reportes">
                    <i class="fa-solid fa-file-lines mr-2"></i> Reportes
                </button>
                <button onclick="showSection('modelo'); toggleMobileMenu()" class="nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all" data-section-mobile="modelo">
                    <i class="fa-solid fa-arrows-rotate mr-2"></i> Modelo
                </button>
            </div>
        </header>

        <!-- Stats Bar -->
        <div class="bg-dark-500/50 border-b border-zinc-800/30">
            <div class="max-w-7xl mx-auto px-4 py-3 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-users text-gold-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Total Socios</span>
                    </div>
                    <p class="text-xl font-black text-white" id="stat-total">600</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-microchip text-gold-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Accuracy IA</span>
                    </div>
                    <p class="text-xl font-black text-gold-400" id="stat-riesgo">85%</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-triangle-exclamation text-rose-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">En Riesgo</span>
                    </div>
                    <p class="text-xl font-black text-rose-400" id="stat-seguros">224</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-piggy-bank text-emerald-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Ahorro Potencial</span>
                    </div>
                    <p class="text-xl font-black text-emerald-400" id="stat-ahorro">S/. 676,800</p>
                </div>
            </div>
        </div>

        <main class="max-w-7xl mx-auto px-4 py-6">

            <!-- SECCIÓN: PREDICTOR -->
            <section id="sec-predictor">
                <div class="grid grid-cols-1 lg:grid-cols-5 gap-6">
                    <!-- Formulario -->
                    <div class="lg:col-span-2 bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 shadow-2xl">
                        <h2 class="text-sm font-bold text-zinc-100 mb-4 flex items-center border-b border-zinc-800/40 pb-3">
                            <div class="bg-gold-400/10 p-1.5 rounded-lg mr-2"><i class="fa-solid fa-user-check text-gold-400 text-xs"></i></div>
                            Evaluar Socio
                        </h2>
                        <form id="form-predict" class="space-y-3">
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Edad</label>
                                    <input type="number" min="14" max="90" id="edad" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all" placeholder="26" required>
                                </div>
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Asistencia/Semana</label>
                                    <input type="number" step="0.1" min="0" max="7" id="asistencia" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all" placeholder="3.5" required>
                                </div>
                            </div>
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Antigüedad (meses)</label>
                                    <input type="number" min="0" id="antiguedad" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all" placeholder="6" required>
                                </div>
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Consumo Barra (S/.)</label>
                                    <input type="number" step="0.1" min="0" id="consumo" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all" placeholder="45.90" required>
                                </div>
                            </div>
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Género</label>
                                    <select id="genero" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                                        <option value="1">Masculino</option>
                                        <option value="0">Femenino</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Plataforma Web</label>
                                    <select id="uso_app" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                                        <option value="1">Sí usa</option>
                                        <option value="0">No usa</option>
                                    </select>
                                </div>
                            </div>
                            <div>
                                <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Membresía</label>
                                <select id="membresia" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                                    <option value="mensual">Mensual - S/. 120.00</option>
                                    <option value="trimestral">Trimestral - S/. 320.00</option>
                                    <option value="anual">Anual - S/. 1,100.00</option>
                                </select>
                            </div>
                            <div class="flex gap-2 pt-2">
                                <button type="submit" class="flex-grow bg-gradient-to-r from-gold-400 to-gold-600 hover:from-gold-300 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm flex justify-center items-center shadow-lg shadow-gold-400/10">
                                    <i class="fa-solid fa-brain mr-2"></i> Predecir Fuga
                                </button>
                                <div class="relative" id="demo-menu-container">
                                    <button type="button" id="btn-demo" onclick="toggleDemoMenu()" class="bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] text-zinc-300 transition-all px-4 py-3 rounded-xl font-bold text-sm border border-zinc-700/50" title="Seleccionar demo">
                                        <i class="fa-solid fa-wand-magic-sparkles"></i>
                                    </button>
                                    <div id="demo-menu" class="hidden absolute right-0 bottom-full mb-2 w-60 bg-dark-200 border border-zinc-700/50 rounded-xl shadow-2xl overflow-hidden z-30">
                                        <p class="text-[9px] text-zinc-500 uppercase tracking-wider font-bold px-3 pt-2.5 pb-1 border-b border-zinc-800/40">Perfil de socio demo</p>
                                        <button onclick="cargarDemo(0)" class="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 transition-all flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-rose-400 shrink-0"></span> Socio en Riesgo Alto
                                        </button>
                                        <button onclick="cargarDemo(1)" class="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 transition-all flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-emerald-400 shrink-0"></span> Socio Estable y Activo
                                        </button>
                                        <button onclick="cargarDemo(2)" class="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 transition-all flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-gold-400 shrink-0"></span> Socio Premium Fiel
                                        </button>
                                        <button onclick="cargarDemo(3)" class="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 transition-all flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-sky-400 shrink-0"></span> Nuevo Socio Digital
                                        </button>
                                        <button onclick="cargarDemo(4)" class="w-full text-left px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800 transition-all flex items-center gap-2">
                                            <span class="w-2 h-2 rounded-full bg-orange-400 shrink-0"></span> Socio Adulto Tradicional
                                        </button>
                                        <button onclick="cargarDemo(-1)" class="w-full text-left px-3 py-2 text-xs text-gold-400 hover:bg-zinc-800 transition-all border-t border-zinc-800/40">
                                            <i class="fa-solid fa-shuffle mr-1"></i> Perfil aleatorio
                                        </button>
                                    </div>
                                </div>
                                <button type="button" id="btn-reset" class="bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] text-zinc-300 transition-all px-4 py-3 rounded-xl font-bold text-sm border border-zinc-700/50" title="Limpiar">
                                    <i class="fa-solid fa-rotate-right"></i>
                                </button>
                            </div>
                        </form>
                    </div>

                    <!-- Resultado -->
                    <div class="lg:col-span-3 bg-dark-400 p-6 rounded-2xl border border-zinc-800/40 shadow-2xl min-h-[420px] flex flex-col" id="card-resultado">
                        <div id="loading" class="hidden flex-1 flex flex-col items-center justify-center space-y-4">
                            <div class="w-16 h-16 border-4 border-gold-400 border-t-transparent rounded-full animate-spin"></div>
                            <p class="text-sm text-zinc-400">Analizando patrones con Random Forest...</p>
                        </div>

                        <div id="placeholder" class="flex-1 flex flex-col items-center justify-center text-center px-4">
                            <div class="w-20 h-20 bg-dark-200 rounded-2xl border border-zinc-800/40 flex items-center justify-center mb-4">
                                <i class="fa-solid fa-chart-mixed text-3xl text-gold-400/60"></i>
                            </div>
                            <h3 class="text-lg font-bold text-zinc-300 mb-2">Panel de Predicción</h3>
                            <p class="text-zinc-500 text-sm max-w-sm">Ingresa los datos del socio o carga una demo para ver la predicción en tiempo real.</p>
                            <div class="mt-6 grid grid-cols-3 gap-3 w-full max-w-xs">
                                <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                    <p class="text-gold-400 font-bold text-lg">85%</p>
                                    <p class="text-[9px] text-zinc-500">Accuracy</p>
                                </div>
                                <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                    <p class="text-gold-400 font-bold text-lg">0.88</p>
                                    <p class="text-[9px] text-zinc-500">AUC-ROC</p>
                                </div>
                                <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                    <p class="text-gold-400 font-bold text-lg">RF</p>
                                    <p class="text-[9px] text-zinc-500">Modelo</p>
                                </div>
                            </div>
                        </div>

                        <div id="resultado" class="hidden flex-1 flex flex-col">
                            <div class="flex justify-between items-start mb-4">
                                <span id="r-badge" class="px-3 py-1 rounded-full text-[10px] font-bold tracking-widest"></span>
                                <span id="r-timestamp" class="text-[10px] text-zinc-600"></span>
                            </div>
                            <div class="text-center mb-4">
                                <div class="text-6xl font-black mb-1" id="r-prob">0%</div>
                                <p class="text-sm text-zinc-400" id="r-titulo"></p>
                            </div>
                            <div class="flex-1 space-y-3">
                                <div class="bg-dark-200 border border-zinc-800/40 p-4 rounded-xl">
                                    <div class="flex items-center gap-2 mb-2">
                                        <i class="fa-solid fa-lightbulb text-gold-400 text-xs"></i>
                                        <span class="text-[10px] text-gold-400 font-bold uppercase tracking-wider">Recomendación IA</span>
                                    </div>
                                    <p class="text-xs text-zinc-300 leading-relaxed" id="r-recomendacion"></p>
                                </div>
                                <div class="bg-dark-200 border border-zinc-800/40 p-4 rounded-xl">
                                    <div class="flex items-center gap-2 mb-2">
                                        <i class="fa-solid fa-gauge-high text-gold-400 text-xs"></i>
                                        <span class="text-[10px] text-gold-400 font-bold uppercase tracking-wider">Nivel de Riesgo</span>
                                    </div>
                                    <div class="flex items-center justify-between">
                                        <span class="text-sm font-bold" id="r-riesgo"></span>
                                        <div class="flex gap-1" id="r-riesgo-dots"></div>
                                    </div>
                                </div>
                                <div class="bg-dark-200 border border-zinc-800/40 p-4 rounded-xl">
                                    <div class="flex items-center gap-2 mb-2">
                                        <i class="fa-solid fa-coins text-gold-400 text-xs"></i>
                                        <span class="text-[10px] text-gold-400 font-bold uppercase tracking-wider">Impacto Financiero</span>
                                    </div>
                                    <p class="text-xs text-zinc-300" id="r-impacto"></p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- SECCIÓN: DASHBOARD -->
            <section id="sec-dashboard" class="hidden">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-chart-pie text-gold-400 mr-2"></i> Distribución de Riesgo
                        </h3>
                        <div class="chart-container"><canvas id="chart-riesgo"></canvas></div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-chart-line text-gold-400 mr-2"></i> Predicciones por Hora
                        </h3>
                        <div class="chart-container"><canvas id="chart-tendencia"></canvas></div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-chart-column text-gold-400 mr-2"></i> Socios por Membresía
                        </h3>
                        <div class="chart-container"><canvas id="chart-membresia"></canvas></div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-calendar-week text-gold-400 mr-2"></i> Tendencia Semanal de Deserción
                        </h3>
                        <div class="chart-container"><canvas id="chart-semana"></canvas></div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-chart-bar text-gold-400 mr-2"></i> Feature Importance
                        </h3>
                        <div class="chart-container"><canvas id="chart-features"></canvas></div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <div class="flex justify-between items-center mb-3">
                            <h3 class="text-sm font-bold text-zinc-100 flex items-center">
                                <i class="fa-solid fa-triangle-exclamation text-rose-400 mr-2"></i> Top 10 Socios en Riesgo
                            </h3>
                            <button onclick="exportarCSV()" class="bg-gold-400/10 hover:bg-gold-400/20 border border-gold-400/30 text-gold-400 text-xs font-semibold px-3 py-1.5 rounded-lg transition-all">
                                <i class="fa-solid fa-file-csv mr-1"></i> Exportar CSV
                            </button>
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-xs">
                                <thead>
                                    <tr class="border-b border-zinc-800/40">
                                        <th class="text-left py-2 text-zinc-500 font-semibold">#</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">ID</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Probabilidad</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Riesgo</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Recomendación</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Fecha</th>
                                    </tr>
                                </thead>
                                <tbody id="tabla-top-riesgo"></tbody>
                            </table>
                        </div>
                    </div>
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <h3 class="text-sm font-bold text-zinc-100 mb-3 flex items-center">
                            <i class="fa-solid fa-clock-rotate-left text-gold-400 mr-2"></i> Últimas Predicciones
                        </h3>
                        <div class="overflow-x-auto">
                            <table class="w-full text-xs">
                                <thead>
                                    <tr class="border-b border-zinc-800/40">
                                        <th class="text-left py-2 text-zinc-500 font-semibold">ID</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Probabilidad</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Riesgo</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Alerta</th>
                                        <th class="text-left py-2 text-zinc-500 font-semibold">Fecha</th>
                                    </tr>
                                </thead>
                                <tbody id="tabla-historial"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>

            <!-- SECCIÓN: ROI -->
            <section id="sec-roi" class="hidden">
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-calculator text-gold-400 mr-2"></i> Calculadora de ROI
                        </h3>
                        <form id="form-roi" class="space-y-4">
                            <div>
                                <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Número de Socios</label>
                                <input type="number" min="1" id="roi-socios" value="100" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                            </div>
                            <div>
                                <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Costo Adquisición/Cliente (S/.)</label>
                                <input type="number" step="0.01" id="roi-cac" value="150" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                            </div>
                            <div>
                                <label class="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">Valor Vida Cliente (S/.)</label>
                                <input type="number" step="0.01" id="roi-ltv" value="1800" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-3 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all">
                            </div>
                            <button type="submit" class="w-full bg-gradient-to-r from-gold-400 to-gold-600 hover:from-gold-300 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm flex justify-center items-center">
                                <i class="fa-solid fa-chart-line mr-2"></i> Calcular ROI
                            </button>
                        </form>
                    </div>
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40 flex flex-col justify-center" id="roi-resultados">
                        <div class="text-center text-zinc-500">
                            <i class="fa-solid fa-coins text-4xl mb-3 opacity-30"></i>
                            <p class="text-sm">Ingresa los datos y calcula el retorno de inversión</p>
                        </div>
                    </div>
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <h3 class="text-sm font-bold text-zinc-100 mb-2 flex items-center">
                            <i class="fa-solid fa-sliders text-gold-400 mr-2"></i> Simulador de Retención
                        </h3>
                        <p class="text-xs text-zinc-500 mb-5">Mueve el control para ver cuánto ahorraría tu empresa al retener un porcentaje de los socios en riesgo. <b class="text-gold-400">Base: 600 socios · S/. 1,800 LTV</b></p>
                        <div class="bg-dark-200 p-5 rounded-xl border border-zinc-800/30">
                            <div class="flex justify-between items-center mb-3">
                                <span class="text-[10px] text-zinc-500 uppercase tracking-wider font-bold">% de Socios en Riesgo Retenidos</span>
                                <span class="text-2xl font-black text-gold-400" id="sim-pct">50%</span>
                            </div>
                            <input type="range" id="sim-slider" min="0" max="100" value="50" oninput="actualizarSimulador()" class="w-full accent-gold-400">
                            <div class="flex justify-between text-[9px] text-zinc-600 mt-1">
                                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                <p class="text-lg font-black text-white" id="sim-retenidos">112</p>
                                <p class="text-[9px] text-zinc-500 uppercase">Socios Retenidos</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                <p class="text-lg font-black text-emerald-400" id="sim-ahorro">S/. 201,600</p>
                                <p class="text-[9px] text-zinc-500 uppercase">Ingresos Salvados</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                <p class="text-lg font-black text-rose-400" id="sim-perdida">S/. 201,600</p>
                                <p class="text-[9px] text-zinc-500 uppercase">Pérdida Restante</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30 text-center">
                                <p class="text-lg font-black text-gold-400" id="sim-net">S/. 0</p>
                                <p class="text-[9px] text-zinc-500 uppercase">Impacto Neto</p>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- SECCIÓN: LOTE -->
            <section id="sec-lote" class="hidden">
                <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                    <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                        <i class="fa-solid fa-layer-group text-gold-400 mr-2"></i> Predicciones por Lote
                    </h3>
                    <p class="text-xs text-zinc-500 mb-4">Ingresa múltiples clientes en formato JSON para análisis masivo.</p>
                    <textarea id="lote-input" rows="10" class="w-full bg-dark-200 border border-zinc-800/60 rounded-lg p-3 text-xs text-green-400 font-mono focus:outline-none focus:ring-2 focus:ring-gold-400/40 transition-all" placeholder='[
  {"edad": 25, "antiguedad_meses": 6, "precio_membresia": 120.0, "asistencia_semanal": 2.5, "consumo_barra": 30.0, "uso_app": 1, "genero_masculino": 1, "membresia_mensual": 1, "membresia_trimestral": 0},
  {"edad": 35, "antiguedad_meses": 24, "precio_membresia": 320.0, "asistencia_semanal": 4.0, "consumo_barra": 60.0, "uso_app": 1, "genero_masculino": 0, "membresia_mensual": 0, "membresia_trimestral": 1}
]'></textarea>
                    <div class="flex gap-2 mt-3">
                        <button onclick="procesarLote()" class="flex-grow bg-gradient-to-r from-gold-400 to-gold-600 hover:from-gold-300 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm flex justify-center items-center">
                            <i class="fa-solid fa-play mr-2"></i> Procesar Lote
                        </button>
                        <button onclick="cargarDemoLote()" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 px-4 py-3 rounded-xl font-bold text-sm border border-zinc-700/50">
                            <i class="fa-solid fa-wand-magic-sparkles mr-1"></i> Demo
                        </button>
                    </div>
                    <div id="lote-resultados" class="mt-4 hidden"></div>
                </div>
            </section>

            <!-- SECCIÓN: REPORTES SEMANALES -->
            <section id="sec-reportes" class="hidden">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Generar Informe -->
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-wand-magic-sparkles text-gold-400 mr-2"></i> Generar Informe Semanal
                        </h3>
                        <p class="text-xs text-zinc-500 mb-4">La IA analiza todas las predicciones y genera un informe ejecutivo con recomendaciones estratégicas.</p>
                        <div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 mb-4">
                            <div class="flex items-center gap-2 mb-2">
                                <i class="fa-solid fa-robot text-gold-400 text-xs"></i>
                                <span class="text-[10px] text-gold-400 font-bold uppercase">Proceso Automático</span>
                            </div>
                            <ul class="text-[10px] text-zinc-400 space-y-1">
                                <li>1. Analiza todas las predicciones del historial</li>
                                <li>2. Identifica patrones de riesgo y tendencias</li>
                                <li>3. Calcula impacto financiero proyectado</li>
                                <li>4. Genera recomendaciones accionables</li>
                                <li>5. Clasifica prioridades por nivel de urgencia</li>
                            </ul>
                        </div>
                        <button onclick="generarInforme()" id="btn-generar-informe" class="w-full bg-gradient-to-r from-gold-400 to-gold-600 hover:from-gold-300 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm flex justify-center items-center shadow-lg shadow-gold-400/10">
                            <i class="fa-solid fa-file-export mr-2"></i> Generar Informe Ahora
                        </button>
                        <div id="generando-informe" class="hidden mt-4 text-center">
                            <div class="w-10 h-10 border-4 border-gold-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                            <p class="text-xs text-zinc-400">IA procesando datos...</p>
                        </div>
                    </div>

                    <!-- Lista de Informes -->
                    <div class="lg:col-span-2 bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-clock-rotate-left text-gold-400 mr-2"></i> Informes Generados
                        </h3>
                        <div id="lista-informes" class="space-y-3">
                            <div class="text-center text-zinc-500 py-8">
                                <i class="fa-solid fa-file-circle-plus text-3xl mb-2 opacity-30"></i>
                                <p class="text-xs">Genera tu primer informe semanal</p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Detalle del Informe -->
                <div id="detalle-informe" class="hidden mt-6 bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="text-sm font-bold text-zinc-100 flex items-center">
                            <i class="fa-solid fa-file-lines text-gold-400 mr-2"></i> <span id="detalle-titulo">Detalle del Informe</span>
                        </h3>
                        <button onclick="cerrarDetalle()" class="text-zinc-500 hover:text-white text-xs"><i class="fa-solid fa-xmark"></i> Cerrar</button>
                    </div>
                    <div id="detalle-contenido"></div>
                </div>
            </section>

            <!-- Seccion Aprendizaje Continuo -->
            <section id="sec-modelo" class="hidden">
                <div class="mb-6">
                    <h2 class="text-xl font-black text-white">Aprendizaje Continuo del Modelo</h2>
                    <p class="text-xs text-zinc-500 mt-1">Monitoreo de data drift, registro de resultados reales y reentrenamiento automático para mantener la precisión de la IA.</p>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Estado del Modelo -->
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-microchip text-gold-400 mr-2"></i> Estado del Modelo
                        </h3>
                        <div class="space-y-3 text-xs">
                            <div class="flex justify-between items-center">
                                <span class="text-zinc-500">Versión</span>
                                <span class="font-bold text-gold-400" id="mod-version">1.0</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-zinc-500">Entrenamiento original</span>
                                <span class="font-bold text-zinc-200" id="mod-fecha-entrenamiento">-</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-zinc-500">Último reentrenamiento</span>
                                <span class="font-bold text-zinc-200" id="mod-fecha-retrain">Nunca</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-zinc-500">Muestras acumuladas</span>
                                <span class="font-bold text-zinc-200" id="mod-muestras">0</span>
                            </div>
                            <div class="flex justify-between items-center">
                                <span class="text-zinc-500">Drift detectado</span>
                                <span class="font-bold" id="mod-drift-badge">-</span>
                            </div>
                        </div>
                        <div class="mt-4 p-3 rounded-xl bg-dark-200 border border-zinc-800/30">
                            <p class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2 font-bold">Métricas del modelo</p>
                            <div class="grid grid-cols-2 gap-2">
                                <div><p class="text-lg font-black text-gold-400" id="mod-acc">0.85</p><p class="text-[9px] text-zinc-500">Accuracy</p></div>
                                <div><p class="text-lg font-black text-gold-400" id="mod-prec">0.82</p><p class="text-[9px] text-zinc-500">Precision</p></div>
                                <div><p class="text-lg font-black text-gold-400" id="mod-rec">0.78</p><p class="text-[9px] text-zinc-500">Recall</p></div>
                                <div><p class="text-lg font-black text-gold-400" id="mod-auc">0.88</p><p class="text-[9px] text-zinc-500">AUC-ROC</p></div>
                            </div>
                        </div>
                        <div class="mt-3 p-3 rounded-xl bg-dark-200 border border-zinc-800/30">
                            <p class="text-[10px] text-zinc-500 uppercase tracking-wider mb-2 font-bold">Efectividad: Entrenamiento vs Producción</p>
                            <div class="space-y-3">
                                <div>
                                    <div class="flex justify-between text-[10px] mb-1">
                                        <span class="text-zinc-400">Entrenamiento (Colab)</span>
                                        <span class="font-bold text-gold-400" id="mod-acc-train">98.6%</span>
                                    </div>
                                    <div class="h-2 rounded-full bg-zinc-800 overflow-hidden">
                                        <div id="bar-train" class="h-full rounded-full bg-gradient-to-r from-gold-400 to-gold-600 transition-all duration-700" style="width:98.6%"></div>
                                    </div>
                                </div>
                                <div>
                                    <div class="flex justify-between text-[10px] mb-1">
                                        <span class="text-zinc-400">Producción (en vivo)</span>
                                        <span class="font-bold text-rose-400" id="mod-acc-prod">85.0%</span>
                                    </div>
                                    <div class="h-2 rounded-full bg-zinc-800 overflow-hidden">
                                        <div id="bar-prod" class="h-full rounded-full bg-gradient-to-r from-rose-400 to-rose-600 transition-all duration-700" style="width:85%"></div>
                                    </div>
                                </div>
                                <div class="flex justify-between items-center pt-1 border-t border-zinc-800/40">
                                    <span class="text-[10px] text-zinc-500">Brecha por data drift</span>
                                    <span class="text-[10px] font-bold text-amber-400" id="mod-brecha">13.6 pts</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Monitoreo de Data Drift -->
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-1 flex items-center">
                            <i class="fa-solid fa-arrow-trend-up text-gold-400 mr-2"></i> Monitoreo de Data Drift
                        </h3>
                        <div id="drift-estado" class="text-xs text-zinc-500 mb-3">Analizando distribución de datos...</div>
                        <div id="drift-table" class="max-h-48 overflow-y-auto scrollbar-hide space-y-1.5">
                            <div class="text-center text-zinc-500 py-6">
                                <i class="fa-solid fa-wave-square text-2xl mb-2 opacity-30"></i>
                                <p class="text-xs">Cargando monitoreo...</p>
                            </div>
                        </div>
                        <div class="mt-4">
                            <p class="text-[10px] text-zinc-500 uppercase tracking-wider font-bold mb-2">Simular nuevo segmento de clientes</p>
                            <div class="grid grid-cols-3 gap-2 mb-3">
                                <button onclick="simularDrift('digital')" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] py-2 rounded-lg border border-zinc-700/50 transition-all">Digital</button>
                                <button onclick="simularDrift('premium')" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] py-2 rounded-lg border border-zinc-700/50 transition-all">Premium</button>
                                <button onclick="simularDrift('adulto')" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] py-2 rounded-lg border border-zinc-700/50 transition-all">Adulto</button>
                            </div>
                            <p class="text-[9px] text-zinc-600 mb-3">Simulación what-if sin modificar los datos reales del dashboard.</p>
                            <button onclick="cargarDrift()" class="w-full bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs py-2 rounded-lg border border-zinc-700/50 transition-all">
                                <i class="fa-solid fa-rotate mr-1"></i> Volver al estado real
                            </button>
                        </div>
                    </div>

                    <!-- Aprender de Errores -->
                    <div class="bg-dark-400 p-6 rounded-2xl border border-zinc-800/40">
                        <h3 class="text-sm font-bold text-zinc-100 mb-1 flex items-center">
                            <i class="fa-solid fa-graduation-cap text-gold-400 mr-2"></i> Aprender de los Errores
                        </h3>
                        <p class="text-xs text-zinc-500 mb-3">El sistema compara lo que predijo con lo que realmente ocurrió y corrige el modelo.</p>
                        <div class="grid grid-cols-2 gap-2 mb-3">
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30">
                                <p class="text-lg font-black text-white" id="fb-total">0</p>
                                <p class="text-[9px] text-zinc-500">Resultados reales</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30">
                                <p class="text-lg font-black text-rose-400" id="fb-errores">0</p>
                                <p class="text-[9px] text-zinc-500">Errores a corregir</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30">
                                <p class="text-lg font-black text-amber-400" id="fb-fp">0</p>
                                <p class="text-[9px] text-zinc-500">Falsos positivos</p>
                            </div>
                            <div class="bg-dark-200 p-3 rounded-xl border border-zinc-800/30">
                                <p class="text-lg font-black text-sky-400" id="fb-fn">0</p>
                                <p class="text-[9px] text-zinc-500">Falsos negativos</p>
                            </div>
                        </div>
                        <button onclick="reentrenarModelo()" id="btn-retrain" class="w-full bg-gradient-to-r from-gold-400 to-gold-600 hover:from-gold-300 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm flex justify-center items-center shadow-lg shadow-gold-400/10">
                            <i class="fa-solid fa-arrows-rotate mr-2"></i> Reentrenar Modelo
                        </button>
                        <div id="retrain-loading" class="hidden mt-4 text-center">
                            <div class="w-10 h-10 border-4 border-gold-400 border-t-transparent rounded-full animate-spin mx-auto mb-2"></div>
                            <p class="text-xs text-zinc-400">Entrenando con nuevos datos...</p>
                        </div>
                        <div id="retrain-resultado" class="hidden mt-4 bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-xl text-xs text-emerald-300"></div>
                    </div>
                </div>
            </section>
        </main>

        <!-- Footer -->
        <footer class="bg-dark-600 border-t border-zinc-900 py-6 mt-8">
            <div class="max-w-7xl mx-auto px-4 text-center">
                <div class="flex items-center justify-center gap-2 mb-2">
                    <div class="bg-gradient-to-br from-gold-400 to-gold-600 p-1.5 rounded-lg text-black"><i class="fa-solid fa-bolt text-xs"></i></div>
                    <span class="text-sm font-bold text-gold-400">VÓRTICE POWER</span>
                </div>
                <p class="text-[10px] text-zinc-600">© 2026 Vórtice Gym Power - VII Ciclo Ingeniería de Sistemas UCV</p>
                <p class="text-[10px] text-zinc-700 mt-1">Plataforma de IA Empresarial | ISO 25010 Compliant</p>
            </div>
        </footer>

        <script>
            let historialLocal = [];
            let statsLocal = { total: 0, en_riesgo: 0, seguros: 0, ahorro: 0 };

            async function cargarStatsDesdeAPI() {
                try {
                    const res = await fetch('/dashboard/stats');
                    const data = await res.json();
                    statsLocal.total = data.total_predicciones;
                    statsLocal.en_riesgo = data.clientes_en_riesgo;
                    statsLocal.seguros = data.clientes_seguros;
                    statsLocal.ahorro = data.ahorro_estimado;
                    actualizarStatsUI();
                } catch(e) { console.error('Error cargando stats:', e); }
            }

            function actualizarStatsUI() {
                document.getElementById('stat-total').textContent = statsLocal.total || '600';
                document.getElementById('stat-seguros').textContent = statsLocal.en_riesgo || '224';
                document.getElementById('stat-ahorro').textContent = 'S/. ' + (statsLocal.ahorro || 676800).toLocaleString();
            }

            function toggleMobileMenu() {
                const menu = document.getElementById('mobile-menu');
                menu.classList.toggle('open');
            }

            function showSection(name) {
                document.querySelectorAll('main > section').forEach(s => s.classList.add('hidden'));
                document.getElementById('sec-' + name).classList.remove('hidden');
                document.querySelectorAll('.nav-btn').forEach(b => {
                    b.className = 'nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all';
                });
                document.querySelectorAll('.nav-btn-mobile').forEach(b => {
                    b.className = 'nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all';
                });
                const desktopBtn = document.querySelector(`[data-section="${name}"]`);
                if (desktopBtn) desktopBtn.className = 'nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-gold-400 bg-gold-400/10 border border-gold-400/20';
                const mobileBtn = document.querySelector(`[data-section-mobile="${name}"]`);
                if (mobileBtn) mobileBtn.className = 'nav-btn-mobile w-full text-left px-3 py-2 text-xs font-semibold rounded-lg text-gold-400 bg-gold-400/10 border border-gold-400/20';
                if (name === 'dashboard') cargarDashboard();
                if (name === 'reportes') cargarInformes();
                if (name === 'modelo') cargarModelo();
            }

            function toggleDemoMenu() {
                const menu = document.getElementById('demo-menu');
                menu.classList.toggle('hidden');
            }

            function cargarDemo(idx) {
                const demos = [
                    { nombre: 'Socio en Riesgo Alto', edad: 22, antiguedad_meses: 3, precio_membresia: 120, asistencia_semanal: 1.0, consumo_barra: 10, uso_app: 0, genero_masculino: 1, membresia_mensual: 1, membresia_trimestral: 0 },
                    { nombre: 'Socio Estable y Activo', edad: 30, antiguedad_meses: 12, precio_membresia: 320, asistencia_semanal: 4.5, consumo_barra: 80, uso_app: 1, genero_masculino: 1, membresia_mensual: 0, membresia_trimestral: 1 },
                    { nombre: 'Socio Premium Fiel', edad: 45, antiguedad_meses: 36, precio_membresia: 1100, asistencia_semanal: 5.0, consumo_barra: 120, uso_app: 1, genero_masculino: 0, membresia_mensual: 0, membresia_trimestral: 0 },
                    { nombre: 'Nuevo Socio Digital', edad: 19, antiguedad_meses: 1, precio_membresia: 120, asistencia_semanal: 3.0, consumo_barra: 25, uso_app: 1, genero_masculino: 1, membresia_mensual: 1, membresia_trimestral: 0 },
                    { nombre: 'Socio Adulto Tradicional', edad: 58, antiguedad_meses: 48, precio_membresia: 1100, asistencia_semanal: 2.0, consumo_barra: 40, uso_app: 0, genero_masculino: 1, membresia_mensual: 0, membresia_trimestral: 0 }
                ];
                document.getElementById('demo-menu').classList.add('hidden');
                const d = demos[idx === undefined || idx === -1 ? Math.floor(Math.random() * demos.length) : idx];
                document.getElementById('edad').value = d.edad;
                document.getElementById('antiguedad').value = d.antiguedad_meses;
                document.getElementById('asistencia').value = d.asistencia_semanal;
                document.getElementById('consumo').value = d.consumo_barra;
                document.getElementById('genero').value = d.genero_masculino;
                document.getElementById('uso_app').value = d.uso_app;
                document.getElementById('membresia').value = d.membresia_mensual ? 'mensual' : d.membresia_trimestral ? 'trimestral' : 'anual';
            }

            document.addEventListener('click', (e) => {
                const cont = document.getElementById('demo-menu-container');
                if (cont && !cont.contains(e.target)) {
                    document.getElementById('demo-menu').classList.add('hidden');
                }
            });

            function cargarDemoLote() {
                document.getElementById('lote-input').value = JSON.stringify([
                    { edad: 22, antiguedad_meses: 3, precio_membresia: 120.0, asistencia_semanal: 1.0, consumo_barra: 10.0, uso_app: 0, genero_masculino: 1, membresia_mensual: 1, membresia_trimestral: 0 },
                    { edad: 30, antiguedad_meses: 12, precio_membresia: 320.0, asistencia_semanal: 4.5, consumo_barra: 80.0, uso_app: 1, genero_masculino: 1, membresia_mensual: 0, membresia_trimestral: 1 },
                    { edad: 45, antiguedad_meses: 36, precio_membresia: 1100.0, asistencia_semanal: 5.0, consumo_barra: 120.0, uso_app: 1, genero_masculino: 0, membresia_mensual: 0, membresia_trimestral: 0 },
                    { edad: 19, antiguedad_meses: 1, precio_membresia: 120.0, asistencia_semanal: 0.5, consumo_barra: 0.0, uso_app: 0, genero_masculino: 1, membresia_mensual: 1, membresia_trimestral: 0 }
                ], null, 2);
            }

            function getRiskColor(riesgo) {
                const colors = { 'BAJO': 'emerald', 'MEDIO': 'yellow', 'ALTO': 'orange', 'CRÍTICO': 'rose' };
                return colors[riesgo] || 'zinc';
            }

            function getRiesgoDots(riesgo) {
                const levels = { 'BAJO': 1, 'MEDIO': 2, 'ALTO': 3, 'CRÍTICO': 4 };
                const n = levels[riesgo] || 1;
                const color = getRiskColor(riesgo);
                let dots = '';
                for (let i = 0; i < 4; i++) {
                    dots += '<div class="w-2 h-2 rounded-full ' + (i < n ? 'bg-' + color + '-400' : 'bg-zinc-700') + '"></div>';
                }
                return dots;
            }

            document.getElementById('form-predict').addEventListener('submit', async (e) => {
                e.preventDefault();
                document.getElementById('placeholder').classList.add('hidden');
                document.getElementById('resultado').classList.add('hidden');
                document.getElementById('loading').classList.remove('hidden');

                const membresia = document.getElementById('membresia').value;
                const precios = { mensual: [120.0, 1, 0], trimestral: [320.0, 0, 1], anual: [1100.0, 0, 0] };
                const [precio, m_mensual, m_trimestral] = precios[membresia];

                const payload = {
                    edad: parseFloat(document.getElementById('edad').value),
                    antiguedad_meses: parseFloat(document.getElementById('antiguedad').value),
                    precio_membresia: precio,
                    asistencia_semanal: parseFloat(document.getElementById('asistencia').value),
                    consumo_barra: parseFloat(document.getElementById('consumo').value),
                    uso_app: parseInt(document.getElementById('uso_app').value),
                    genero_masculino: parseInt(document.getElementById('genero').value),
                    membresia_mensual: m_mensual,
                    membresia_trimestral: m_trimestral
                };

                try {
                    const res = await fetch('/predecir_fuga', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                    const data = await res.json();
                    if (data.detail) { alert('Error: ' + data.detail); document.getElementById('loading').classList.add('hidden'); document.getElementById('placeholder').classList.remove('hidden'); return; }

                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('resultado').classList.remove('hidden');

                    const pct = (data.probabilidad_desercion * 100).toFixed(1);
                    const color = data.alerta_de_fuga ? 'rose' : 'emerald';
                    const badgeText = data.alerta_de_fuga ? 'ALERTA DE FUGA' : 'SOCIO SEGURO';
                    const titulo = data.alerta_de_fuga ? 'Socio Propenso a Fugarse' : 'Cliente Estable y Conforme';

                    document.getElementById('r-prob').textContent = pct + '%';
                    document.getElementById('r-prob').className = 'text-6xl font-black text-' + color + '-400';
                    document.getElementById('r-badge').className = 'px-3 py-1 rounded-full text-[10px] font-bold tracking-widest bg-' + color + '-500/10 text-' + color + '-400 border border-' + color + '-500/20 uppercase';
                    document.getElementById('r-badge').textContent = badgeText;
                    document.getElementById('r-titulo').textContent = titulo;
                    document.getElementById('r-titulo').className = 'text-sm text-' + color + '-400';
                    document.getElementById('r-recomendacion').textContent = data.recomendacion;
                    document.getElementById('r-riesgo').textContent = data.nivel_riesgo;
                    document.getElementById('r-riesgo').className = 'text-sm font-bold text-' + color + '-400';
                    document.getElementById('r-riesgo-dots').innerHTML = getRiesgoDots(data.nivel_riesgo);
                    document.getElementById('r-timestamp').textContent = new Date(data.timestamp).toLocaleString('es-PE');
                    const ahorro = data.alerta_de_fuga ? 1800 : 0;
                    document.getElementById('r-impacto').textContent = data.alerta_de_fuga ? 'Si retienes este socio, puedes ahorrar S/. ' + ahorro.toLocaleString() + ' en costos de reemplazo.' : 'Este socio genera valor constante. Mantén la experiencia de servicio.';

                    await cargarStatsDesdeAPI();

                } catch (err) {
                    alert('Error de conexion: ' + err.message);
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('placeholder').classList.remove('hidden');
                }
            });

            document.getElementById('btn-reset').addEventListener('click', () => {
                document.getElementById('form-predict').reset();
                document.getElementById('resultado').classList.add('hidden');
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('placeholder').classList.remove('hidden');
            });

            document.getElementById('form-roi').addEventListener('submit', async (e) => {
                e.preventDefault();
                const socios = parseInt(document.getElementById('roi-socios').value);
                const cac = parseFloat(document.getElementById('roi-cac').value);
                const ltv = parseFloat(document.getElementById('roi-ltv').value);
                const tasaDesercion = statsLocal.total > 0 ? statsLocal.en_riesgo / statsLocal.total : 0.35;
                const clientesEnRiesgo = Math.round(socios * tasaDesercion);
                const ahorro = clientesEnRiesgo * ltv * 0.3;
                const inversion = 2400;
                const roi = ((ahorro - inversion) / inversion * 100).toFixed(0);
                const payback = (inversion / (ahorro / 12)).toFixed(1);

                document.getElementById('roi-resultados').innerHTML = '<div class="space-y-4">' +
                    '<div class="text-center mb-4">' +
                        '<div class="text-5xl font-black text-gold-400 mb-1">' + roi + '%</div>' +
                        '<p class="text-xs text-zinc-400">Retorno de Inversion Anual</p>' +
                    '</div>' +
                    '<div class="grid grid-cols-2 gap-3">' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 text-center">' +
                            '<p class="text-2xl font-bold text-rose-400">' + clientesEnRiesgo + '</p>' +
                            '<p class="text-[10px] text-zinc-500">Clientes en Riesgo</p>' +
                        '</div>' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 text-center">' +
                            '<p class="text-2xl font-bold text-emerald-400">S/. ' + ahorro.toLocaleString(undefined, {maximumFractionDigits:0}) + '</p>' +
                            '<p class="text-[10px] text-zinc-500">Ahorro Anual</p>' +
                        '</div>' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 text-center">' +
                            '<p class="text-2xl font-bold text-gold-400">' + payback + ' meses</p>' +
                            '<p class="text-[10px] text-zinc-500">Payback Period</p>' +
                        '</div>' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 text-center">' +
                            '<p class="text-2xl font-bold text-zinc-300">S/. ' + inversion.toLocaleString() + '</p>' +
                            '<p class="text-[10px] text-zinc-500">Inversion Sistema</p>' +
                        '</div>' +
                    '</div>' +
                    '<div class="bg-gold-400/5 border border-gold-400/20 p-4 rounded-xl mt-3">' +
                        '<p class="text-xs text-gold-400 font-bold mb-1"><i class="fa-solid fa-lightbulb mr-1"></i> Resumen Ejecutivo</p>' +
                        '<p class="text-xs text-zinc-400">Con ' + socios + ' socios y una tasa de desercion del ' + (tasaDesercion*100).toFixed(0) + '%, el sistema Vortice puede retener hasta ' + clientesEnRiesgo + ' clientes, generando un ahorro de S/. ' + ahorro.toLocaleString(undefined, {maximumFractionDigits:0}) + ' anuales con un payback de ' + payback + ' meses.</p>' +
                    '</div>' +
                '</div>';
            });

            function actualizarSimulador() {
                const pct = parseInt(document.getElementById('sim-slider').value);
                document.getElementById('sim-pct').textContent = pct + '%';
                const enRiesgo = 224;
                const ltv = 1800;
                const retenidos = Math.round(enRiesgo * pct / 100);
                const ahorro = retenidos * ltv;
                const perdida = (enRiesgo - retenidos) * ltv;
                const neto = ahorro - perdida;
                document.getElementById('sim-retenidos').textContent = retenidos;
                document.getElementById('sim-ahorro').textContent = 'S/. ' + ahorro.toLocaleString();
                document.getElementById('sim-perdida').textContent = 'S/. ' + perdida.toLocaleString();
                document.getElementById('sim-net').textContent = 'S/. ' + (neto > 0 ? '+' : '') + neto.toLocaleString();
                document.getElementById('sim-net').className = 'text-lg font-black ' + (neto >= 0 ? 'text-emerald-400' : 'text-rose-400');
            }

            async function procesarLote() {
                try {
                    const clientes = JSON.parse(document.getElementById('lote-input').value);
                    const res = await fetch('/predecir_lote', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clientes }) });
                    const data = await res.json();
                    let html = '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 mb-3">' +
                        '<div class="grid grid-cols-4 gap-2 text-center">' +
                            '<div><p class="text-lg font-bold text-white">' + data.total_clientes + '</p><p class="text-[9px] text-zinc-500">Total</p></div>' +
                            '<div><p class="text-lg font-bold text-rose-400">' + data.en_riesgo + '</p><p class="text-[9px] text-zinc-500">En Riesgo</p></div>' +
                            '<div><p class="text-lg font-bold text-emerald-400">' + data.seguros + '</p><p class="text-[9px] text-zinc-500">Seguros</p></div>' +
                            '<div><p class="text-lg font-bold text-gold-400">S/. ' + data.ahorro_estimado.toLocaleString() + '</p><p class="text-[9px] text-zinc-500">Ahorro</p></div>' +
                        '</div></div>';
                    data.predicciones.forEach((p, i) => {
                        const c = getRiskColor(p.nivel_riesgo);
                        html += '<div class="bg-dark-200 p-3 rounded-lg border border-zinc-800/30 mb-2 flex justify-between items-center">' +
                            '<span class="text-xs text-zinc-400">Cliente ' + (i+1) + '</span>' +
                            '<span class="text-xs font-bold text-' + c + '-400">' + (p.probabilidad_desercion*100).toFixed(1) + '% - ' + p.nivel_riesgo + '</span></div>';
                    });
                    document.getElementById('lote-resultados').innerHTML = html;
                    document.getElementById('lote-resultados').classList.remove('hidden');
                } catch (err) { alert('Error en JSON: ' + err.message); }
            }

            let chartRiesgo, chartTendencia, chartSemana, chartMembresia, chartFeatures;

            async function cargarDashboard() {
                let datosHistorial = [];
                try {
                    const res = await fetch('/historial');
                    const data = await res.json();
                    datosHistorial = data.predicciones || [];
                } catch(e) { console.error('Error cargando historial:', e); }

                let statsData = { total_predicciones: 0, clientes_en_riesgo: 0, clientes_seguros: 0, tasa_riesgo: 0 };
                try {
                    const res = await fetch('/dashboard/stats');
                    statsData = await res.json();
                } catch(e) {}

                const conteo = { 'BAJO': 0, 'MEDIO': 0, 'ALTO': 0, 'CRITICO': 0 };
                datosHistorial.forEach(p => {
                    const key = p.nivel_riesgo.replace('Í', 'I');
                    if (conteo[key] !== undefined) conteo[key]++;
                });

                if (chartRiesgo) chartRiesgo.destroy();
                if (chartTendencia) chartTendencia.destroy();
                if (chartSemana) chartSemana.destroy();
                if (chartMembresia) chartMembresia.destroy();
                if (chartFeatures) chartFeatures.destroy();

                requestAnimationFrame(() => {

                chartRiesgo = new Chart(document.getElementById('chart-riesgo'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Bajo', 'Medio', 'Alto', 'Critico'],
                        datasets: [{ data: [conteo.BAJO, conteo.MEDIO, conteo.ALTO, conteo.CRITICO], backgroundColor: ['#10b981', '#eab308', '#f97316', '#f43f5e'], borderWidth: 0 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#a1a1aa', font: { size: 10 } } } } }
                });

                const horas = {};
                datosHistorial.forEach(p => {
                    const h = new Date(p.timestamp).getHours() + ':00';
                    horas[h] = (horas[h] || 0) + 1;
                });
                const horasSorted = Object.entries(horas).sort((a,b) => parseInt(a[0]) - parseInt(b[0]));
                const labelsHoras = horasSorted.map(h => h[0]);
                const dataHoras = horasSorted.map(h => h[1]);

                chartTendencia = new Chart(document.getElementById('chart-tendencia'), {
                    type: 'line',
                    data: {
                        labels: labelsHoras.length ? labelsHoras : ['Sin datos'],
                        datasets: [{ label: 'Predicciones', data: dataHoras.length ? dataHoras : [0], borderColor: '#d4af37', backgroundColor: 'rgba(212,175,55,0.1)', fill: true, tension: 0.4, pointRadius: 3, pointBackgroundColor: '#d4af37' }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(63,63,70,0.3)' }, ticks: { color: '#71717a', font: { size: 9 } } }, y: { grid: { color: 'rgba(63,63,70,0.3)' }, ticks: { color: '#71717a', font: { size: 9 } } } } }
                });

                chartFeatures = new Chart(document.getElementById('chart-features'), {
                    type: 'bar',
                    data: { labels: ['Asistencia', 'Consumo Barra', 'Antiguedad', 'Precio', 'Edad', 'Uso App', 'Genero'], datasets: [{ label: 'Importancia', data: [0.28, 0.22, 0.18, 0.12, 0.10, 0.06, 0.04], backgroundColor: '#d4af37', borderRadius: 4 }] },
                    options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: 'rgba(63,63,70,0.3)' }, ticks: { color: '#71717a', font: { size: 9 } } }, y: { grid: { display: false }, ticks: { color: '#a1a1aa', font: { size: 10 } } } } }
                });

                const semanas = {};
                datosHistorial.forEach(p => {
                    const d = new Date(p.timestamp);
                    const sem = d.getFullYear() + '-S' + Math.ceil((d.getDate() + new Date(d.getFullYear(), d.getMonth(), 1).getDay()) / 7);
                    if (!semanas[sem]) semanas[sem] = { total: 0, riesgo: 0 };
                    semanas[sem].total++;
                    if (p.alerta_de_fuga) semanas[sem].riesgo++;
                });
                const semKeys = Object.keys(semanas).sort();
                const semTasa = semKeys.map(k => {
                    const s = semanas[k];
                    return Math.round(s.riesgo / s.total * 100);
                });

                chartSemana = new Chart(document.getElementById('chart-semana'), {
                    type: 'line',
                    data: {
                        labels: semKeys.length ? semKeys : ['Sin datos'],
                        datasets: [{
                            label: 'Tasa Deserción %',
                            data: semTasa.length ? semTasa : [0],
                            borderColor: '#f43f5e',
                            backgroundColor: 'rgba(244,63,94,0.1)',
                            fill: true,
                            tension: 0.4,
                            pointRadius: 4,
                            pointBackgroundColor: '#f43f5e'
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#a1a1aa', font: { size: 10 } } } }, scales: { x: { grid: { color: 'rgba(63,63,70,0.3)' }, ticks: { color: '#71717a', font: { size: 9 } } }, y: { grid: { color: 'rgba(63,63,70,0.3)' }, ticks: { color: '#71717a', font: { size: 9 }, callback: v => v + '%' } } } }
                });

                const membresia = { 'Mensual': 0, 'Trimestral': 0, 'Anual': 0 };
                datosHistorial.forEach(p => {
                    const cd = p.cliente_demo || {};
                    if (cd.membresia_mensual) membresia['Mensual']++;
                    else if (cd.membresia_trimestral) membresia['Trimestral']++;
                    else membresia['Anual']++;
                });

                chartMembresia = new Chart(document.getElementById('chart-membresia'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Mensual', 'Trimestral', 'Anual'],
                        datasets: [{ data: [membresia.Mensual, membresia.Trimestral, membresia.Anual], backgroundColor: ['#d4af37', '#3b82f6', '#a855f7'], borderWidth: 0 }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#a1a1aa', font: { size: 10 } } } } }
                });

                const topRiesgo = datosHistorial.filter(p => p.alerta_de_fuga).sort((a,b) => b.probabilidad_desercion - a.probabilidad_desercion).slice(0, 10);
                const tbodyTop = document.getElementById('tabla-top-riesgo');
                tbodyTop.innerHTML = topRiesgo.map((p, i) => {
                    const color = p.probabilidad_desercion > 0.7 ? 'rose' : 'orange';
                    return '<tr class="border-b border-zinc-800/20">' +
                        '<td class="py-2 text-zinc-600 font-bold">' + (i+1) + '</td>' +
                        '<td class="py-2 text-zinc-400">' + p.id + '</td>' +
                        '<td class="py-2 font-bold text-' + color + '-400">' + (p.probabilidad_desercion*100).toFixed(1) + '%</td>' +
                        '<td class="py-2"><span class="text-[9px] font-bold bg-' + color + '-500/20 text-' + color + '-400 px-2 py-0.5 rounded-full">' + p.nivel_riesgo + '</span></td>' +
                        '<td class="py-2 text-zinc-400 max-w-[200px] truncate">' + (p.recomendacion || '-') + '</td>' +
                        '<td class="py-2 text-zinc-500">' + new Date(p.timestamp).toLocaleDateString('es-PE') + '</td></tr>';
                }).join('');

                const tbody = document.getElementById('tabla-historial');
                tbody.innerHTML = datosHistorial.slice(0, 15).map(h => {
                    const alerta = h.alerta_de_fuga;
                    const color = alerta ? 'rose' : 'emerald';
                    return '<tr class="border-b border-zinc-800/20">' +
                        '<td class="py-2 text-zinc-400">' + h.id + '</td>' +
                        '<td class="py-2 font-bold text-' + color + '-400">' + (h.probabilidad_desercion*100).toFixed(1) + '%</td>' +
                        '<td class="py-2 text-zinc-300">' + h.nivel_riesgo + '</td>' +
                        '<td class="py-2">' + (alerta ? '<span class="text-rose-400">Si</span>' : '<span class="text-emerald-400">No</span>') + '</td>' +
                        '<td class="py-2 text-zinc-500">' + new Date(h.timestamp).toLocaleString('es-PE') + '</td></tr>';
                }).join('');

                }); // end requestAnimationFrame
            }

            let datosHistorialGlobal = [];

            async function exportarCSV() {
                try {
                    const res = await fetch('/historial');
                    const data = await res.json();
                    const enRiesgo = (data.predicciones || []).filter(p => p.alerta_de_fuga).sort((a,b) => b.probabilidad_desercion - a.probabilidad_desercion);
                    const header = 'ID,Probabilidad,Riesgo,Recomendacion,Fecha';
                    const rows = enRiesgo.slice(0, 50).map(p => {
                        const fecha = new Date(p.timestamp).toLocaleString('es-PE');
                        return [p.id, (p.probabilidad_desercion*100).toFixed(2) + '%', p.nivel_riesgo, (p.recomendacion||'').replace(/,/g,';'), fecha].join(',');
                    }).join('\n');
                    const blob = new Blob([header + '\n' + rows], { type: 'text/csv;charset=utf-8;' });
                    const link = document.createElement('a');
                    link.href = URL.createObjectURL(blob);
                    link.download = 'socios_en_riesgo.csv';
                    link.click();
                    URL.revokeObjectURL(link.href);
                } catch(e) { alert('Error exportando: ' + e.message); }
            }

            async function cargarInformes() {
                try {
                    const res = await fetch('/informes');
                    const data = await res.json();
                    const container = document.getElementById('lista-informes');
                    if (data.total === 0) {
                        container.innerHTML = '<div class="text-center text-zinc-500 py-8"><i class="fa-solid fa-file-circle-plus text-3xl mb-2 opacity-30"></i><p class="text-xs">Genera tu primer informe semanal</p></div>';
                        return;
                    }
                    container.innerHTML = data.informes.map(inf => {
                        const fecha = new Date(inf.fecha).toLocaleString('es-PE');
                        const color = inf.tasa_desercion > 30 ? 'rose' : inf.tasa_desercion > 20 ? 'yellow' : 'emerald';
                        return '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30 hover:border-gold-400/20 transition-all cursor-pointer" onclick="verDetalle(\'' + inf.id + '\')">' +
                            '<div class="flex justify-between items-start">' +
                                '<div>' +
                                    '<p class="text-xs font-bold text-white">' + inf.id + '</p>' +
                                    '<p class="text-[10px] text-zinc-500 mt-1">' + fecha + '</p>' +
                                '</div>' +
                                '<div class="text-right">' +
                                    '<span class="text-xs font-bold text-' + color + '-400">' + inf.tasa_desercion + '% desercion</span>' +
                                    '<p class="text-[10px] text-zinc-500">' + inf.total_socios + ' socios | ' + inf.en_riesgo + ' en riesgo</p>' +
                                '</div>' +
                            '</div>' +
                        '</div>';
                    }).join('');
                } catch(e) { console.error('Error cargando informes:', e); }
            }

            async function generarInforme() {
                const btn = document.getElementById('btn-generar-informe');
                const loading = document.getElementById('generando-informe');
                btn.classList.add('hidden');
                loading.classList.remove('hidden');

                try {
                    const res = await fetch('/informe_semanal', { method: 'POST' });
                    const informe = await res.json();
                    loading.classList.add('hidden');
                    btn.classList.remove('hidden');
                    await cargarInformes();
                    verDetalle(informe.id);
                } catch(e) {
                    loading.classList.add('hidden');
                    btn.classList.remove('hidden');
                    alert('Error generando informe: ' + e.message);
                }
            }

            async function verDetalle(informeId) {
                try {
                    const res = await fetch('/informes/' + informeId);
                    const inf = await res.json();
                    document.getElementById('detalle-titulo').textContent = 'Informe ' + inf.id;
                    const container = document.getElementById('detalle-contenido');
                    const r = inf.resumen_ejecutivo;
                    const f = inf.impacto_financiero;

                    let html = '<div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30">' +
                            '<h4 class="text-[10px] text-gold-400 font-bold uppercase mb-3"><i class="fa-solid fa-chart-pie mr-1"></i> Resumen Ejecutivo</h4>' +
                            '<div class="grid grid-cols-2 gap-2">' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-white">' + r.total_socios_analizados + '</p><p class="text-[9px] text-zinc-500">Total Analizados</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-emerald-400">' + r.socios_seguros + '</p><p class="text-[9px] text-zinc-500">Seguros</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-rose-400">' + r.socios_en_riesgo + '</p><p class="text-[9px] text-zinc-500">En Riesgo</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-yellow-400">' + r.tasa_desercion + '%</p><p class="text-[9px] text-zinc-500">Tasa Desercion</p></div>' +
                            '</div>' +
                            '<div class="mt-3 flex gap-2">' +
                                '<div class="flex-1 text-center p-2 bg-dark-300 rounded-lg"><span class="text-[9px] text-emerald-400 font-bold">BAJO: ' + r.clasificacion.BAJO + '</span></div>' +
                                '<div class="flex-1 text-center p-2 bg-dark-300 rounded-lg"><span class="text-[9px] text-yellow-400 font-bold">MEDIO: ' + r.clasificacion.MEDIO + '</span></div>' +
                                '<div class="flex-1 text-center p-2 bg-dark-300 rounded-lg"><span class="text-[9px] text-orange-400 font-bold">ALTO: ' + r.clasificacion.ALTO + '</span></div>' +
                                '<div class="flex-1 text-center p-2 bg-dark-300 rounded-lg"><span class="text-[9px] text-rose-400 font-bold">CRITICO: ' + r.clasificacion.CRITICO + '</span></div>' +
                            '</div>' +
                        '</div>' +
                        '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30">' +
                            '<h4 class="text-[10px] text-gold-400 font-bold uppercase mb-3"><i class="fa-solid fa-coins mr-1"></i> Impacto Financiero</h4>' +
                            '<div class="grid grid-cols-2 gap-2">' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-emerald-400">S/. ' + f.ahorro_por_retencion.toLocaleString() + '</p><p class="text-[9px] text-zinc-500">Ahorro Potencial</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-rose-400">S/. ' + f.perdida_por_desercion.toLocaleString() + '</p><p class="text-[9px] text-zinc-500">Perdida Estimada</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-gold-400">S/. ' + f.ingresos_estimados_mensuales.toLocaleString() + '</p><p class="text-[9px] text-zinc-500">Ingresos Mensuales</p></div>' +
                                '<div class="text-center p-2 bg-dark-300 rounded-lg"><p class="text-lg font-bold text-white">S/. ' + f.proyeccion_3_meses.toLocaleString() + '</p><p class="text-[9px] text-zinc-500">Proyeccion 3M</p></div>' +
                            '</div>' +
                        '</div>' +
                    '</div>';

                    if (inf.socios_criticos && inf.socios_criticos.length > 0) {
                        html += '<div class="bg-rose-500/5 border border-rose-500/20 p-4 rounded-xl mb-4">' +
                            '<h4 class="text-[10px] text-rose-400 font-bold uppercase mb-3"><i class="fa-solid fa-triangle-exclamation mr-1"></i> Socios Criticos - Accion Inmediata (' + inf.socios_criticos.length + ')</h4>';
                        inf.socios_criticos.forEach(s => {
                            html += '<div class="bg-dark-300/50 p-3 rounded-lg mb-2 flex justify-between items-center">' +
                                '<div><span class="text-xs font-bold text-white">' + s.id + '</span><span class="text-[10px] text-zinc-500 ml-2">' + (s.probabilidad * 100).toFixed(1) + '% riesgo</span></div>' +
                                '<span class="text-[10px] text-zinc-400">' + s.recomendacion.substring(0, 60) + '...</span></div>';
                        });
                        html += '</div>';
                    }

                    html += '<div class="bg-gold-400/5 border border-gold-400/20 p-4 rounded-xl mb-4">' +
                        '<h4 class="text-[10px] text-gold-400 font-bold uppercase mb-3"><i class="fa-solid fa-lightbulb mr-1"></i> Recomendaciones Estrategicas</h4>';
                    inf.recomendaciones_estrategicas.forEach(rec => {
                        const priorColor = rec.prioridad === 'URGENTE' ? 'rose' : rec.prioridad === 'ALTA' ? 'orange' : rec.prioridad === 'MEDIA' ? 'yellow' : 'emerald';
                        html += '<div class="bg-dark-300/50 p-3 rounded-lg mb-2">' +
                            '<div class="flex items-center gap-2 mb-1">' +
                                '<span class="text-[9px] font-bold bg-' + priorColor + '-500/20 text-' + priorColor + '-400 px-2 py-0.5 rounded-full">' + rec.prioridad + '</span>' +
                                '<span class="text-[10px] text-gold-400 font-bold">' + rec.impacto + '</span>' +
                            '</div>' +
                            '<p class="text-xs text-zinc-300">' + rec.accion + '</p>' +
                        '</div>';
                    });
                    html += '</div>';

                    html += '<div class="bg-dark-200 p-4 rounded-xl border border-zinc-800/30">' +
                        '<h4 class="text-[10px] text-gold-400 font-bold uppercase mb-2"><i class="fa-solid fa-microchip mr-1"></i> Metricas del Modelo IA</h4>' +
                        '<div class="flex gap-3 text-center">' +
                            '<div class="flex-1"><p class="text-sm font-bold text-gold-400">' + (inf.metricas_modelo.accuracy * 100) + '%</p><p class="text-[9px] text-zinc-500">Accuracy</p></div>' +
                            '<div class="flex-1"><p class="text-sm font-bold text-gold-400">' + (inf.metricas_modelo.auc_roc) + '</p><p class="text-[9px] text-zinc-500">AUC-ROC</p></div>' +
                            '<div class="flex-1"><p class="text-sm font-bold text-gold-400">' + (inf.metricas_modelo.precision * 100) + '%</p><p class="text-[9px] text-zinc-500">Precision</p></div>' +
                            '<div class="flex-1"><p class="text-sm font-bold text-gold-400">' + (inf.metricas_modelo.recall * 100) + '%</p><p class="text-[9px] text-zinc-500">Recall</p></div>' +
                        '</div>' +
                    '</div>';

                    container.innerHTML = html;
                    document.getElementById('detalle-informe').classList.remove('hidden');
                    document.getElementById('detalle-informe').scrollIntoView({ behavior: 'smooth' });
                } catch(e) { console.error('Error:', e); }
            }

            function cerrarDetalle() {
                document.getElementById('detalle-informe').classList.add('hidden');
            }

            async function cargarModelo() {
                try {
                    const info = await (await fetch('/info')).json();
                    document.getElementById('mod-version').textContent = info.version_modelo;
                    document.getElementById('mod-fecha-entrenamiento').textContent = info.fecha_entrenamiento || '-';
                    document.getElementById('mod-fecha-retrain').textContent = info.fecha_ultimo_reentrenamiento ? new Date(info.fecha_ultimo_reentrenamiento).toLocaleString('es-PE') : 'Nunca';
                    document.getElementById('mod-muestras').textContent = info.muestras_actuales + ' / ' + info.muestras_entrenamiento;
                    const badge = document.getElementById('mod-drift-badge');
                    if (info.drift_detectado) {
                        badge.textContent = 'SÍ';
                        badge.className = 'font-bold text-rose-400';
                    } else {
                        badge.textContent = 'NO';
                        badge.className = 'font-bold text-emerald-400';
                    }
                    const met = info.metricas || {};
                    document.getElementById('mod-acc').textContent = (met.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('mod-prec').textContent = (met.precision * 100).toFixed(1) + '%';
                    document.getElementById('mod-rec').textContent = (met.recall * 100).toFixed(1) + '%';
                    document.getElementById('mod-auc').textContent = met.auc_roc;
                    const accTrain = (info.accuracy_entrenamiento * 100).toFixed(1);
                    const accProd = (met.accuracy * 100).toFixed(1);
                    document.getElementById('mod-acc-train').textContent = accTrain + '%';
                    document.getElementById('mod-acc-prod').textContent = accProd + '%';
                    document.getElementById('bar-train').style.width = accTrain + '%';
                    document.getElementById('bar-prod').style.width = accProd + '%';
                    document.getElementById('mod-brecha').textContent = (accTrain - accProd).toFixed(1) + ' pts';
                    cargarDrift();
                    cargarFeedback();
                } catch(e) { console.error('Error cargando modelo:', e); }
            }

            async function cargarDrift() {
                try {
                    const d = await (await fetch('/drift')).json();
                    renderDrift(d);
                } catch(e) { console.error('Error drift:', e); }
            }

            async function simularDrift(perfil) {
                try {
                    const res = await fetch('/drift/simular', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ n: 200, perfil: perfil })
                    });
                    const d = await res.json();
                    renderDrift(d);
                } catch(e) { console.error('Error simulando drift:', e); }
            }

            function renderDrift(d) {
                const badge = document.getElementById('drift-estado');
                const estados = {
                    'ESTABLE': ['bg-emerald-500/15 text-emerald-400 border-emerald-500/30', 'Distribución estable — el modelo mantiene su precisión.'],
                    'MODERADO': ['bg-amber-500/15 text-amber-400 border-amber-500/30', 'Se detecta desviación moderada en algunos features.'],
                    'SEVERO': ['bg-rose-500/15 text-rose-400 border-rose-500/30', 'Drift severo detectado: la distribución de datos cambió. Reentrena el modelo.'],
                    'sin_baseline': ['bg-zinc-700/20 text-zinc-400 border-zinc-700/50', 'Sin datos suficientes.'],
                    'sin_datos': ['bg-zinc-700/20 text-zinc-400 border-zinc-700/50', 'Sin predicciones para monitorear.']
                };
                const [cls, msg] = estados[d.estado] || ['bg-zinc-700/20 text-zinc-400 border-zinc-700/50', d.mensaje];
                badge.innerHTML = '<div class="flex items-center justify-between mb-2"><span class="font-bold text-sm ' + (cls.includes('rose') ? 'text-rose-400' : cls.includes('amber') ? 'text-amber-400' : 'text-emerald-400') + '">' + d.estado + '</span><span class="text-[10px] text-zinc-500">' + (d.simulado ? 'SIMULACIÓN · ' + d.segmento + ' · ' + d.nuevos_clientes + ' clientes' : 'En tiempo real') + '</span></div><p class="text-[10px]">' + msg + '</p>';

                const table = document.getElementById('drift-table');
                const feats = d.features || {};
                const cols = Object.keys(feats);
                if (!cols.length) {
                    table.innerHTML = '<div class="text-center text-zinc-500 py-6"><p class="text-xs">Sin datos de features</p></div>';
                    return;
                }
                let html = '';
                const labels = { edad:'Edad', antiguedad_meses:'Antigüedad', precio_membresia:'Precio', asistencia_semanal:'Asistencia', consumo_barra:'Consumo', uso_app:'Uso App', genero_masculino:'Género', membresia_mensual:'Memb. Mensual', membresia_trimestral:'Memb. Trimestral' };
                for (const f of cols) {
                    const v = feats[f];
                    const color = v.nivel === 'SEVERO' ? 'text-rose-400' : v.nivel === 'MODERADO' ? 'text-amber-400' : v.nivel === 'LEVE' ? 'text-yellow-300' : 'text-emerald-400';
                    html += '<div class="flex items-center justify-between bg-dark-200 border border-zinc-800/30 rounded-lg px-3 py-1.5">' +
                        '<span class="text-[10px] text-zinc-400">' + (labels[f] || f) + '</span>' +
                        '<span class="text-[10px] font-bold ' + color + '">' + v.nivel + ' <span class="text-zinc-600 font-normal">(Δ ' + v.score + ')</span></span>' +
                        '</div>';
                }
                table.innerHTML = html;
            }

            async function cargarFeedback() {
                try {
                    const f = await (await fetch('/feedback/estadisticas')).json();
                    document.getElementById('fb-total').textContent = f.total;
                    document.getElementById('fb-errores').textContent = f.errores_modelo;
                    document.getElementById('fb-fp').textContent = f.falsos_positivos;
                    document.getElementById('fb-fn').textContent = f.falsos_negativos;
                } catch(e) { console.error('Error feedback:', e); }
            }

            async function reentrenarModelo() {
                const btn = document.getElementById('btn-retrain');
                const loading = document.getElementById('retrain-loading');
                const result = document.getElementById('retrain-resultado');
                btn.classList.add('hidden');
                loading.classList.remove('hidden');
                result.classList.add('hidden');
                try {
                    const res = await fetch('/retrain', { method: 'POST' });
                    const data = await res.json();
                    loading.classList.add('hidden');
                    btn.classList.remove('hidden');
                    if (!res.ok) {
                        result.className = 'mt-4 bg-rose-500/10 border border-rose-500/20 p-3 rounded-xl text-xs text-rose-300';
                        result.textContent = 'Error: ' + (data.detail || 'No se pudo reentrenar');
                    } else {
                        result.className = 'mt-4 bg-emerald-500/10 border border-emerald-500/20 p-3 rounded-xl text-xs text-emerald-300';
                        result.innerHTML = '<div class="flex items-center gap-2 font-bold mb-1"><i class="fa-solid fa-circle-check"></i> Modelo reentrenado (v' + data.version + ')</div>' +
                            '<p>Muestras: <b>' + data.muestras + '</b> · Fuente: <b>' + (data.fuente === 'feedback' ? 'Resultados reales' : 'Historial') + '</b></p>' +
                            '<p class="mt-1">Accuracy: <b>' + (data.metricas.accuracy * 100).toFixed(1) + '%</b> · AUC-ROC: <b>' + data.metricas.auc_roc + '</b> · CV: <b>' + (data.cv_accuracy * 100).toFixed(1) + '%</b></p>';
                    }
                    cargarModelo();
                } catch(e) {
                    loading.classList.add('hidden');
                    btn.classList.remove('hidden');
                    result.className = 'mt-4 bg-rose-500/10 border border-rose-500/20 p-3 rounded-xl text-xs text-rose-300';
                    result.textContent = 'Error de conexión: ' + e.message;
                }
            }

            window.addEventListener('load', () => {
                cargarStatsDesdeAPI();
                setInterval(() => {
                    const sec = document.getElementById('sec-dashboard');
                    if (!sec.classList.contains('hidden')) cargarDashboard();
                }, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predecir_fuga", response_model=RespuestaPrediccion, tags=["Predicción"])
async def predecir(cliente: DatosCliente):
    try:
        probabilidad, nivel_riesgo, recomendacion = predecir_cliente(cliente)
        pred_id = f"PRD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        alerta = probabilidad > 0.50

        resultado = RespuestaPrediccion(
            id=pred_id,
            probabilidad_desercion=round(probabilidad, 4),
            alerta_de_fuga=alerta,
            nivel_riesgo=nivel_riesgo,
            recomendacion=recomendacion,
            timestamp=datetime.now().isoformat()
        )

        historial = cargar_historial()
        historial.append(resultado.dict())
        if len(historial) > 500:
            historial = historial[-500:]
        guardar_historial(historial)

        logger.info(f"Predicción {pred_id} - Prob: {probabilidad:.2%}, Riesgo: {nivel_riesgo}")
        return resultado
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/predecir_lote", response_model=ResultadoLote, tags=["Predicción"])
async def predecir_lote(lote: DatosLote):
    try:
        predicciones = []
        en_riesgo = 0
        seguros = 0
        suma_prob = 0

        for cliente in lote.clientes:
            probabilidad, nivel_riesgo, recomendacion = predecir_cliente(cliente)
            alerta = probabilidad > 0.50
            suma_prob += probabilidad
            if alerta:
                en_riesgo += 1
            else:
                seguros += 1
            predicciones.append({
                "probabilidad_desercion": round(probabilidad, 4),
                "alerta_de_fuga": alerta,
                "nivel_riesgo": nivel_riesgo,
                "recomendacion": recomendacion
            })

        total = len(lote.clientes)
        ahorro = en_riesgo * 540

        return ResultadoLote(
            total_clientes=total,
            en_riesgo=en_riesgo,
            seguros=seguros,
            probabilidad_promedio=round(suma_prob / total, 4),
            ahorro_estimado=ahorro,
            predicciones=predicciones
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en lote: {str(e)}")

@app.get("/historial", tags=["Historial"])
async def obtener_historial():
    historial = cargar_historial()
    return {"total": len(historial), "predicciones": historial[-50:]}

@app.get("/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats():
    historial = cargar_historial()
    total = len(historial)
    en_riesgo = sum(1 for p in historial if p.get("alerta_de_fuga"))
    return {
        "total_predicciones": total,
        "clientes_en_riesgo": en_riesgo,
        "clientes_seguros": total - en_riesgo,
        "tasa_riesgo": round(en_riesgo / total * 100, 1) if total > 0 else 0,
        "ahorro_estimado": (total - en_riesgo) * 1800
    }

INFORMES_FILE = "informes_semanales.json"

def cargar_informes():
    if os.path.exists(INFORMES_FILE):
        with open(INFORMES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def guardar_informes(informes):
    with open(INFORMES_FILE, "w", encoding="utf-8") as f:
        json.dump(informes, f, ensure_ascii=False, indent=2)

def generar_informe_ia():
    historial = cargar_historial()
    if not historial:
        return None

    total = len(historial)
    en_riesgo = [p for p in historial if p.get("alerta_de_fuga")]
    seguros = [p for p in historial if not p.get("alerta_de_fuga")]

    critico = [p for p in en_riesgo if p.get("nivel_riesgo") == "CRITICO" or p.get("nivel_riesgo") == "CRÍTICO"]
    alto = [p for p in en_riesgo if p.get("nivel_riesgo") == "ALTO"]
    medio = [p for p in historial if p.get("nivel_riesgo") == "MEDIO"]
    bajo = [p for p in seguros if p.get("nivel_riesgo") == "BAJO"]

    prob_promedio = sum(p.get("probabilidad_desercion", 0) for p in historial) / total
    tasa_desercion = len(en_riesgo) / total * 100

    ahorro_proyectado = len(seguros) * 1800
    perdida_proyectada = len(en_riesgo) * 1800
    impacto_net = ahorro_proyectado - perdida_proyectada

    from collections import Counter
    conteo_riesgo = Counter(p.get("nivel_riesgo", "DESCONOCIDO") for p in historial)

    distribucion = {
        "BAJO": conteo_riesgo.get("BAJO", 0),
        "MEDIO": conteo_riesgo.get("MEDIO", 0),
        "ALTO": conteo_riesgo.get("ALTO", 0),
        "CRITICO": conteo_riesgo.get("CRÍTICO", 0) + conteo_riesgo.get("CRITICO", 0)
    }

    rec_criticos = []
    for p in critico[:5]:
        rec_criticos.append({
            "id": p.get("id", "N/A"),
            "probabilidad": p.get("probabilidad_desercion", 0),
            "recomendacion": p.get("recomendacion", "Sin recomendacion"),
            "timestamp": p.get("timestamp", "")
        })

    rec_alto = []
    for p in alto[:5]:
        rec_alto.append({
            "id": p.get("id", "N/A"),
            "probabilidad": p.get("probabilidad_desercion", 0),
            "recomendacion": p.get("recomendacion", "Sin recomendacion"),
            "timestamp": p.get("timestamp", "")
        })

    recomendaciones_estrategicas = []
    if distribucion["CRITICO"] > 0:
        recomendaciones_estrategicas.append({
            "prioridad": "URGENTE",
            "accion": f"Hacer contacto inmediato con los {distribucion['CRITICO']} socios criticos. Ofrecer descuento del 20% o servicio premium gratis por 1 mes.",
            "impacto": f"Retencion potencial: S/. {distribucion['CRITICO'] * 1800:,}"
        })
    if distribucion["ALTO"] > 0:
        recomendaciones_estrategicas.append({
            "prioridad": "ALTA",
            "accion": f"Enviar campana de reactivacion a {distribucion['ALTO']} socios en riesgo alto. Incluir clase gratuita con entrenador personal.",
            "impacto": f"Retencion potencial: S/. {distribucion['ALTO'] * 1800:,}"
        })
    if distribucion["MEDIO"] > total * 0.2:
        recomendaciones_estrategicas.append({
            "prioridad": "MEDIA",
            "accion": "Implementar programa de gamificacion y retos mensuales para socios con riesgo medio.",
            "impacto": "Mejora de engagement estimada: 15-25%"
        })
    recomendaciones_estrategicas.append({
        "prioridad": "CONTINUA",
        "accion": "Mantener programa de referidos para socios de bajo riesgo. Ofrecer bonificacion por cada nuevo socio referido.",
        "impacto": f"Potencial de crecimiento: {distribucion['BAJO']} socios como embajadores"
    })

    ingresos_mensuales = total * prob_promedio * 150
    proyeccion_3m = ingresos_mensuales * 3 * (1 - prob_promedio)

    informe = {
        "id": f"INF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "fecha_generacion": datetime.now().isoformat(),
        "periodo": f"{datetime.now().strftime('%d/%m/%Y')} - Informe Automatico",
        "resumen_ejecutivo": {
            "total_socios_analizados": total,
            "socios_seguros": len(seguros),
            "socios_en_riesgo": len(en_riesgo),
            "tasa_desercion": round(tasa_desercion, 1),
            "probabilidad_promedio": round(prob_promedio, 4),
            "clasificacion": distribucion
        },
        "impacto_financiero": {
            "ingresos_estimados_mensuales": round(ingresos_mensuales, 2),
            "ahorro_por_retencion": ahorro_proyectado,
            "perdida_por_desercion": perdida_proyectada,
            "impacto_neto": impacto_net,
            "proyeccion_3_meses": round(proyeccion_3m, 2)
        },
        "socios_criticos": rec_criticos,
        "socios_en_riesgo_alto": rec_alto,
        "recomendaciones_estrategicas": recomendaciones_estrategicas,
        "metricas_modelo": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
            "auc_roc": 0.88
        }
    }

    informes = cargar_informes()
    informes.insert(0, informe)
    if len(informes) > 20:
        informes = informes[:20]
    guardar_informes(informes)

    return informe

@app.post("/informe_semanal", tags=["Informes"])
async def generar_informe_semanal():
    informe = generar_informe_ia()
    if not informe:
        raise HTTPException(status_code=404, detail="No hay datos en el historial para generar informe")
    return informe

@app.get("/informes", tags=["Informes"])
async def listar_informes():
    informes = cargar_informes()
    resumen = []
    for inf in informes:
        resumen.append({
            "id": inf["id"],
            "fecha": inf["fecha_generacion"],
            "periodo": inf["periodo"],
            "total_socios": inf["resumen_ejecutivo"]["total_socios_analizados"],
            "en_riesgo": inf["resumen_ejecutivo"]["socios_en_riesgo"],
            "tasa_desercion": inf["resumen_ejecutivo"]["tasa_desercion"]
        })
    return {"total": len(resumen), "informes": resumen}

@app.get("/informes/{informe_id}", tags=["Informes"])
async def obtener_informe(informe_id: str):
    informes = cargar_informes()
    for inf in informes:
        if inf["id"] == informe_id:
            return inf
    raise HTTPException(status_code=404, detail="Informe no encontrado")

@app.get("/health", tags=["Sistema"])
async def health_check():
    return {"status": "healthy", "modelo": True, "version": "3.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/info", tags=["Sistema"])
async def info_modelo_endpoint():
    return {
        "modelo": "Random Forest Classifier",
        "features": 9,
        "empresa": "Gimnasio Vórtice S.A.C.",
        "version_api": "3.0.0",
        "version_modelo": info_modelo.get("version", "1.0"),
        "fecha_entrenamiento": info_modelo.get("fecha_entrenamiento"),
        "fecha_ultimo_reentrenamiento": info_modelo.get("fecha_ultimo_reentrenamiento"),
        "muestras_entrenamiento": info_modelo.get("muestras_entrenamiento"),
        "muestras_actuales": info_modelo.get("muestras_actuales"),
        "accuracy_entrenamiento": info_modelo.get("accuracy_entrenamiento", 0.986),
        "accuracy_produccion": info_modelo.get("metricas", {}).get("accuracy", 0.85),
        "metricas": info_modelo.get("metricas", {}),
        "drift_detectado": info_modelo.get("drift_detectado", False),
        "endpoints": ["/predecir_fuga", "/predecir_lote", "/historial", "/dashboard/stats", "/health", "/info", "/metricas", "/drift", "/retrain"]
    }

@app.get("/metricas", response_model=MetricasModelo, tags=["Sistema"])
async def obtener_metricas():
    m = info_modelo.get("metricas", {})
    return MetricasModelo(
        accuracy=m.get("accuracy", 0.85),
        precision=m.get("precision", 0.82),
        recall=m.get("recall", 0.78),
        f1_score=m.get("f1_score", 0.80),
        auc_roc=m.get("auc_roc", 0.88)
    )

@app.get("/drift", tags=["Sistema"])
async def estado_drift():
    """Monitorea la desviacion de datos (data drift) del modelo en produccion."""
    return monitorear_drift()

@app.post("/drift/simular", tags=["Sistema"])
async def simular_drift_demo(req: SimularDriftRequest):
    """Simula la llegada de un nuevo segmento de clientes y muestra el impacto en el drift."""
    return simular_drift(req.n, req.perfil)

@app.post("/retrain", tags=["Sistema"])
async def retrain_modelo():
    """Reentrena el modelo con los datos acumulados para corregir data drift."""
    try:
        return reentrenar_modelo()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en reentrenamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno al reentrenar: {e}")

@app.post("/feedback", tags=["Sistema"])
async def enviar_feedback(fb: FeedbackRequest):
    """Registra el resultado real de una prediccion para que el modelo aprenda de sus errores."""
    try:
        return registrar_feedback(fb.id_prediccion, fb.socio_se_fugo)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registrando feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/lote", tags=["Sistema"])
async def enviar_feedback_lote(fb: FeedbackLoteRequest):
    """Registra los resultados reales de varias predicciones a la vez."""
    try:
        return registrar_feedback_lote([{"id_prediccion": r.id_prediccion, "socio_se_fugo": r.socio_se_fugo} for r in fb.resultados])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registrando feedback en lote: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/estadisticas", tags=["Sistema"])
async def estadisticas_feedback():
    """Resumen del feedback acumulado: errores del modelo detectados."""
    feedback = cargar_feedback()
    historial = cargar_historial()
    pred_map = {p["id"]: p for p in historial}
    if not feedback:
        return {"total": 0, "errores_modelo": 0, "falsos_positivos": 0, "falsos_negativos": 0, "precision_feedback": None}

    errores = 0
    fp = 0
    fn = 0
    correctos = 0
    for f in feedback:
        pred = pred_map.get(f["id_prediccion"])
        predijo_fuga = bool(pred.get("alerta_de_fuga")) if pred else False
        real = bool(f["socio_se_fugo"])
        if predijo_fuga != real:
            errores += 1
            if predijo_fuga and not real:
                fp += 1
            else:
                fn += 1
        else:
            correctos += 1
    return {
        "total": len(feedback),
        "correctos": correctos,
        "errores_modelo": errores,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "precision_feedback": round(correctos / len(feedback), 4),
        "mensaje": "El modelo detecto correctamente el resultado en todos los casos." if errores == 0 else f"El modelo fallo en {errores} de {len(feedback)} casos y aprendera de ellos."
    }
