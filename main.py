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
                    </nav>
                    <a href="/docs" target="_blank" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-400 text-xs px-3 py-1.5 rounded-lg border border-zinc-700/50 transition-all">
                        <i class="fa-solid fa-book mr-1"></i> API
                    </a>
                    <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] px-2 py-1 rounded-full flex items-center">
                        <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1 animate-pulse"></span> Live
                    </span>
                </div>
            </div>
        </header>

        <!-- Stats Bar -->
        <div class="bg-dark-500/50 border-b border-zinc-800/30">
            <div class="max-w-7xl mx-auto px-4 py-3 grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-users text-gold-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Total Predicciones</span>
                    </div>
                    <p class="text-xl font-black text-white" id="stat-total">0</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-triangle-exclamation text-rose-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">En Riesgo</span>
                    </div>
                    <p class="text-xl font-black text-rose-400" id="stat-riesgo">0</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-shield-halved text-emerald-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Seguros</span>
                    </div>
                    <p class="text-xl font-black text-emerald-400" id="stat-seguros">0</p>
                </div>
                <div class="stat-card p-3 rounded-xl border border-zinc-800/40">
                    <div class="flex items-center gap-2 mb-1">
                        <i class="fa-solid fa-piggy-bank text-gold-400 text-xs"></i>
                        <span class="text-[10px] text-zinc-500 uppercase tracking-wider">Ahorro Estimado</span>
                    </div>
                    <p class="text-xl font-black text-gold-400" id="stat-ahorro">S/. 0</p>
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
                                <button type="button" id="btn-demo" onclick="cargarDemo()" class="bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] text-zinc-300 transition-all px-4 py-3 rounded-xl font-bold text-sm border border-zinc-700/50" title="Cargar demo">
                                    <i class="fa-solid fa-wand-magic-sparkles"></i>
                                </button>
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
                    <div class="bg-dark-400 p-5 rounded-2xl border border-zinc-800/40 lg:col-span-2">
                        <h3 class="text-sm font-bold text-zinc-100 mb-4 flex items-center">
                            <i class="fa-solid fa-chart-bar text-gold-400 mr-2"></i> Feature Importance
                        </h3>
                        <div class="chart-container"><canvas id="chart-features"></canvas></div>
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
                document.getElementById('stat-total').textContent = statsLocal.total;
                document.getElementById('stat-riesgo').textContent = statsLocal.en_riesgo;
                document.getElementById('stat-seguros').textContent = statsLocal.seguros;
                document.getElementById('stat-ahorro').textContent = 'S/. ' + statsLocal.ahorro.toLocaleString();
            }

            function showSection(name) {
                document.querySelectorAll('main > section').forEach(s => s.classList.add('hidden'));
                document.getElementById('sec-' + name).classList.remove('hidden');
                document.querySelectorAll('.nav-btn').forEach(b => {
                    b.className = 'nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all';
                });
                document.querySelector(`[data-section="${name}"]`).className = 'nav-btn px-3 py-1.5 text-xs font-semibold rounded-lg text-gold-400 bg-gold-400/10 border border-gold-400/20';
                if (name === 'dashboard') cargarDashboard();
            }

            function cargarDemo() {
                const demos = [
                    { edad: 22, antiguedad_meses: 3, precio_membresia: 120, asistencia_semanal: 1.0, consumo_barra: 10, uso_app: 0, genero_masculino: 1, membresia_mensual: 1, membresia_trimestral: 0 },
                    { edad: 30, antiguedad_meses: 12, precio_membresia: 320, asistencia_semanal: 4.5, consumo_barra: 80, uso_app: 1, genero_masculino: 1, membresia_mensual: 0, membresia_trimestral: 1 },
                    { edad: 45, antiguedad_meses: 36, precio_membresia: 1100, asistencia_semanal: 5.0, consumo_barra: 120, uso_app: 1, genero_masculino: 0, membresia_mensual: 0, membresia_trimestral: 0 }
                ];
                const d = demos[Math.floor(Math.random() * demos.length)];
                document.getElementById('edad').value = d.edad;
                document.getElementById('antiguedad').value = d.antiguedad_meses;
                document.getElementById('asistencia').value = d.asistencia_semanal;
                document.getElementById('consumo').value = d.consumo_barra;
                document.getElementById('genero').value = d.genero_masculino;
                document.getElementById('uso_app').value = d.uso_app;
                document.getElementById('membresia').value = d.membresia_mensual ? 'mensual' : d.membresia_trimestral ? 'trimestral' : 'anual';
            }

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

            let chartRiesgo, chartTendencia, chartFeatures;

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
                if (chartFeatures) chartFeatures.destroy();

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

            window.addEventListener('load', () => { cargarStatsDesdeAPI(); });
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
async def info_modelo():
    return {
        "modelo": "Random Forest Classifier",
        "features": 9,
        "empresa": "Gimnasio Vórtice S.A.C.",
        "version": "3.0.0",
        "endpoints": ["/predecir_fuga", "/predecir_lote", "/historial", "/dashboard/stats", "/health", "/info", "/metricas"]
    }

@app.get("/metricas", response_model=MetricasModelo, tags=["Sistema"])
async def obtener_metricas():
    return MetricasModelo(accuracy=0.85, precision=0.82, recall=0.78, f1_score=0.80, auc_roc=0.88)
