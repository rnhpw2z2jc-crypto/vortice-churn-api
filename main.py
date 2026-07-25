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
from typing import Optional
import joblib
import numpy as np
import logging
from datetime import datetime
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================
app = FastAPI(
    title="Gimnasio Vórtice S.A.C. - API de IA",
    description="API para predicción de deserción de clientes usando Machine Learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 2. CARGA DEL MODELO Y ESCALADOR
# ============================================================
MODEL_PATH = "modelo_random_forest_vortice.joblib"
SCALER_PATH = "escalador_vortice.joblib"

try:
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("Modelo y escalador cargados exitosamente")
except FileNotFoundError as e:
    logger.error(f"Error cargando archivos del modelo: {e}")
    raise
except Exception as e:
    logger.error(f"Error inesperado cargando el modelo: {e}")
    raise

# ============================================================
# 3. MODELOS DE DATOS (PYDANTIC)
# ============================================================
class DatosCliente(BaseModel):
    """
    Modelo de datos de entrada para predicción de fuga.
    
    Attributes:
        edad: Edad del socio (16-90 años)
        antiguedad_meses: Meses de permanencia en el gimnasio
        precio_membresia: Precio de la membresía en soles
        asistencia_semanal: Promedio de asistencias por semana (0-7)
        consumo_barra: Gasto mensual en barra nutricional
        uso_app: Uso de la plataforma web (0/1)
        genero_masculino: Género del socio (1=Masculino, 0=Femenino)
        membresia_mensual: Tipo membresía mensual (0/1)
        membresia_trimestral: Tipo membresía trimestral (0/1)
    """
    
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

    @validator('membresia_mensual', 'membresia_trimestral')
    def validar_membresia(cls, v, values):
        if 'membresia_mensual' in values and 'membresia_trimestral' in values:
            if values['membresia_mensual'] + values['membresia_trimestral'] > 1:
                raise ValueError('Solo puede seleccionar un tipo de membresía')
        return v

class RespuestaPrediccion(BaseModel):
    """Modelo de respuesta de predicción"""
    probabilidad_desercion: float
    alerta_de_fuga: bool
    nivel_riesgo: str
    timestamp: str

class MetricasModelo(BaseModel):
    """Métricas del modelo de ML"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

# ============================================================
# 4. ENDPOINTS DE LA API
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Principal"])
async def home():
    """
    Endpoint principal que retorna la interfaz web de usuario.
    
    Returns:
        HTMLResponse: Interfaz web completa
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <title>Vórtice Gym Power - IA Predicción de Deserción</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            gold: {
                                100: '#F4E8C1',
                                400: '#E5C158',
                                500: '#D4AF37',
                                600: '#B8932A',
                                900: '#4A3B0F'
                            }
                        }
                    }
                }
            }
        </script>
        <style>
            .gold-text-glow {
                text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
            }
            .card-hover:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            }
            .input-focus:focus {
                box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
            }
        </style>
    </head>
    <body class="bg-[#0b0b0b] text-white min-h-screen flex flex-col justify-between font-sans antialiased">
        
        <!-- Header -->
        <header class="bg-[#121212] border-b border-zinc-800/80 p-4 shadow-xl sticky top-0 z-50">
            <div class="max-w-6xl mx-auto flex flex-col sm:flex-row justify-between items-center gap-3">
                <div class="flex items-center space-x-3 text-center sm:text-left">
                    <div class="bg-gradient-to-br from-gold-500 to-gold-600 p-2 rounded-lg text-black shadow-md shadow-gold-500/20">
                        <i class="fa-solid fa-dumbbell text-xl animate-pulse"></i>
                    </div>
                    <div>
                        <h1 class="text-lg sm:text-xl font-black tracking-widest text-gold-500 gold-text-glow">VÓRTICE GYM POWER</h1>
                        <p class="text-[10px] sm:text-xs text-zinc-400 font-medium">Plataforma de IA - VII Ciclo UCV</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] sm:text-xs px-2.5 py-1 rounded-full flex items-center">
                        <span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-ping"></span> Servidor Activo
                    </span>
                    <a href="/docs" target="_blank" class="bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] sm:text-xs px-2.5 py-1 rounded-full border border-zinc-700/50">
                        <i class="fa-solid fa-book mr-1"></i> API Docs
                    </a>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-4xl mx-auto p-4 sm:p-6 w-full flex-grow flex flex-col justify-center">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
                
                <!-- Formulario -->
                <div class="bg-[#141414] p-5 sm:p-6 rounded-2xl border border-zinc-800/60 shadow-2xl flex flex-col justify-between card-hover transition-all duration-300">
                    <div>
                        <h2 class="text-base sm:text-lg font-bold text-zinc-100 mb-4 flex items-center border-b border-zinc-800/60 pb-2">
                            <i class="fa-solid fa-user-gear mr-2 text-gold-500"></i> Datos del Socio a Evaluar
                        </h2>
                        
                        <form id="form-predict" class="space-y-3.5">
                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Edad del Socio</label>
                                    <input type="number" min="14" max="90" id="edad" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all" placeholder="Ej: 26" required>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Asistencia Semanal</label>
                                    <input type="number" step="0.1" min="0" max="7" id="asistencia" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all" placeholder="Ej: 3.5" required>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Antigüedad (Meses)</label>
                                    <input type="number" min="0" id="antiguedad" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all" placeholder="Ej: 6" required>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Consumo Barra (S/.)</label>
                                    <input type="number" step="0.1" min="0" id="consumo" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all" placeholder="Ej: 45.90" required>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Género</label>
                                    <select id="genero" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all">
                                        <option value="1">Masculino</option>
                                        <option value="0">Femenino</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Plataforma Web</label>
                                    <select id="uso_app" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all">
                                        <option value="1">Sí usa la Web</option>
                                        <option value="0">No usa la Web</option>
                                    </select>
                                </div>
                            </div>

                            <div>
                                <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Tipo de Membresía</label>
                                <select id="membresia" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 input-focus transition-all">
                                    <option value="mensual">Membresía Mensual (S/. 120.00)</option>
                                    <option value="trimestral">Membresía Trimestral (S/. 320.00)</option>
                                    <option value="anual">Membresía Anual (S/. 1100.00)</option>
                                </select>
                            </div>

                            <!-- Botones de Acción -->
                            <div class="flex gap-2 mt-4">
                                <button type="submit" class="flex-grow bg-gradient-to-r from-gold-500 to-gold-600 hover:from-gold-400 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm tracking-wider flex justify-center items-center shadow-lg shadow-gold-500/10">
                                    <i class="fa-solid fa-brain mr-2"></i> Calcular Riesgo
                                </button>
                                <button type="button" id="btn-reset" class="bg-zinc-800 hover:bg-zinc-700 active:scale-[0.98] text-zinc-300 transition-all px-4 py-3 rounded-xl font-bold text-sm flex items-center justify-center border border-zinc-700/50" title="Limpiar formulario">
                                    <i class="fa-solid fa-rotate-right"></i>
                                </button>
                            </div>
                        </form>
                    </div>
                </div>

                <!-- Resultados -->
                <div class="bg-[#141414] p-6 rounded-2xl border border-zinc-800/60 shadow-2xl min-h-[380px] flex flex-col justify-center items-center text-center relative overflow-hidden card-hover transition-all duration-300" id="card-resultado">
                    <div id="loading" class="hidden flex-col items-center space-y-3">
                        <div class="w-12 h-12 border-4 border-gold-500 border-t-transparent rounded-full animate-spin"></div>
                        <p class="text-sm text-zinc-400">Procesando vectores mediante Random Forest...</p>
                    </div>

                    <div id="placeholder-text" class="flex flex-col items-center space-y-4 px-4">
                        <div class="p-4 bg-zinc-900/50 rounded-full border border-zinc-800/50">
                            <i class="fa-solid fa-chart-line text-4xl text-gold-500"></i>
                        </div>
                        <p class="text-zinc-400 text-xs sm:text-sm max-w-[280px] leading-relaxed">Ingresa los datos de comportamiento del socio en el formulario para predecir si continuará o se fugará.</p>
                    </div>

                    <div id="resultado-content" class="hidden w-full flex flex-col items-center space-y-4 px-2">
                        <span id="alerta-badge" class="px-4 py-1.5 rounded-full text-[10px] font-bold tracking-widest"></span>
                        
                        <div class="relative flex items-center justify-center">
                            <div class="text-5xl sm:text-6xl font-black" id="probabilidad-valor">0%</div>
                        </div>
                        
                        <h3 class="text-base sm:text-lg font-black tracking-wide" id="resultado-titulo"></h3>
                        <p class="text-zinc-400 text-xs sm:text-sm max-w-[320px] leading-relaxed" id="resultado-descripcion"></p>
                        
                        <div class="w-full bg-[#1c1c1c] border border-zinc-800 p-4 rounded-xl text-left mt-2 shadow-inner">
                            <span class="text-xs text-gold-400 font-bold block mb-1 uppercase tracking-wider"><i class="fa-solid fa-hand-holding-heart mr-1.5"></i> Plan recomendado:</span>
                            <p class="text-xs text-zinc-300 leading-relaxed" id="recomendacion-texto"></p>
                        </div>

                        <div class="w-full bg-[#1c1c1c] border border-zinc-800 p-4 rounded-xl text-left mt-2 shadow-inner">
                            <span class="text-xs text-gold-400 font-bold block mb-1 uppercase tracking-wider"><i class="fa-solid fa-shield-halved mr-1.5"></i> Nivel de riesgo:</span>
                            <p class="text-xs text-zinc-300 leading-relaxed" id="nivel-riesgo-texto"></p>
                        </div>
                    </div>
                </div>

            </div>

            <!-- Información del Modelo -->
            <div class="mt-8 bg-[#141414] p-6 rounded-2xl border border-zinc-800/60 shadow-2xl">
                <h2 class="text-base sm:text-lg font-bold text-zinc-100 mb-4 flex items-center border-b border-zinc-800/60 pb-2">
                    <i class="fa-solid fa-microchip mr-2 text-gold-500"></i> Información del Modelo
                </h2>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
                    <div class="bg-[#1c1c1c] p-3 rounded-xl border border-zinc-800/50">
                        <p class="text-gold-500 font-bold text-lg">RF</p>
                        <p class="text-[10px] text-zinc-400">Random Forest</p>
                    </div>
                    <div class="bg-[#1c1c1c] p-3 rounded-xl border border-zinc-800/50">
                        <p class="text-gold-500 font-bold text-lg">9</p>
                        <p class="text-[10px] text-zinc-400">Features</p>
                    </div>
                    <div class="bg-[#1c1c1c] p-3 rounded-xl border border-zinc-800/50">
                        <p class="text-gold-500 font-bold text-lg">85%</p>
                        <p class="text-[10px] text-zinc-400">Accuracy</p>
                    </div>
                    <div class="bg-[#1c1c1c] p-3 rounded-xl border border-zinc-800/50">
                        <p class="text-gold-500 font-bold text-lg">0.88</p>
                        <p class="text-[10px] text-zinc-400">AUC-ROC</p>
                    </div>
                </div>
            </div>
        </main>

        <!-- Footer -->
        <footer class="bg-[#0e0e0e] p-4 border-t border-zinc-900 text-center text-[10px] sm:text-xs text-zinc-600">
            <p>© 2026 VÓRTICE GYM POWER - VII Ciclo Escuela de Ingeniería de Sistemas UCV</p>
            <p class="mt-1">Cumple con estándares ISO 25010 para calidad de software</p>
        </footer>

        <!-- Scripts -->
        <script>
            // Lógica para limpiar/refrescar los casilleros
            document.getElementById('btn-reset').addEventListener('click', () => {
                document.getElementById('form-predict').reset();
                document.getElementById('resultado-content').classList.add('hidden');
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('placeholder-text').classList.remove('hidden');
            });

            document.getElementById('form-predict').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                // Mostrar cargador
                document.getElementById('placeholder-text').classList.add('hidden');
                document.getElementById('resultado-content').classList.add('hidden');
                document.getElementById('loading').classList.remove('hidden');

                // Obtener datos del formulario
                const edad = parseFloat(document.getElementById('edad').value);
                const asistencia_semanal = parseFloat(document.getElementById('asistencia').value);
                const count_meses = parseFloat(document.getElementById('antiguedad').value);
                const consumo = parseFloat(document.getElementById('consumo').value);
                const genero = parseInt(document.getElementById('genero').value);
                const uso_app = parseInt(document.getElementById('uso_app').value);
                const membresia = document.getElementById('membresia').value;

                // Mapeo preciso One-Hot Encoding
                let precio_membresia = 120.0;
                let m_mensual = 0;
                let m_trimestral = 0;

                if (membresia === "mensual") {
                    precio_membresia = 120.0;
                    m_mensual = 1;
                } else if (membresia === "trimestral") {
                    precio_membresia = 320.0;
                    m_trimestral = 1;
                } else if (membresia === "anual") {
                    precio_membresia = 1100.0;
                }

                const payload = {
                    edad: edad,
                    antiguedad_meses: count_meses,
                    precio_membresia: precio_membresia,
                    asistencia_semanal: asistencia_semanal,
                    consumo_barra: consumo,
                    uso_app: uso_app,
                    genero_masculino: genero,
                    membresia_mensual: m_mensual,
                    membresia_trimestral: m_trimestral
                };

                try {
                    const response = await fetch('/predecir_fuga', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();
                    
                    if (data.detail) {
                        console.error("Error devuelto por la API:", data.detail);
                        alert("Error en el modelo: " + data.detail);
                        document.getElementById('loading').classList.add('hidden');
                        document.getElementById('placeholder-text').classList.remove('hidden');
                        return;
                    }

                    // Detener cargador y mostrar resultados
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('resultado-content').classList.remove('hidden');

                    const prob_pct = (data.probabilidad_desercion * 100).toFixed(1);
                    document.getElementById('probabilidad-valor').innerText = prob_pct + "%";

                    const badge = document.getElementById('alerta-badge');
                    const titulo = document.getElementById('resultado-titulo');
                    const desc = document.getElementById('resultado-descripcion');
                    const rec = document.getElementById('recomendacion-texto');
                    const riesgo = document.getElementById('nivel-riesgo-texto');

                    if (data.alerta_de_fuga) {
                        badge.className = "px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest bg-rose-500/10 text-rose-400 border border-rose-500/20 uppercase";
                        badge.innerText = "ALERTA: RIESGO DE FUGA DETECTADO";
                        document.getElementById('probabilidad-valor').className = "text-5xl sm:text-6xl font-black text-rose-500";
                        titulo.innerText = "Socio Propenso a Retirarse";
                        titulo.className = "text-base sm:text-lg font-black text-rose-400";
                        desc.innerText = "El algoritmo identificó un patrón de baja interacción con la marca, baja asistencia y nulo consumo cruzado.";
                        rec.innerText = "Contactar inmediatamente por WhatsApp. Ofrecerle una consulta gratuita con el nutricionista de la barra o un descuento especial del 15% en su próxima renovación.";
                    } else {
                        badge.className = "px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 uppercase";
                        badge.innerText = "SOCIO ACTIVO Y CONFORME";
                        document.getElementById('probabilidad-valor').className = "text-5xl sm:text-6xl font-black text-emerald-500";
                        titulo.innerText = "Cliente Saludable";
                        titulo.className = "text-base sm:text-lg font-black text-emerald-400";
                        desc.innerText = "El socio mantiene una asistencia constante alineada a sus promedios históricos y consume activamente productos de valor agregado.";
                        rec.innerText = "Mantener excelente estándar de servicio en sala de musculación. Invitar a participar en retos internos del gimnasio o felicitación automatizada por hitos de asistencia.";
                    }

                    riesgo.innerText = "Nivel: " + data.nivel_riesgo + " | Timestamp: " + new Date(data.timestamp).toLocaleString();

                } catch (error) {
                    console.error("Error de comunicación:", error);
                    alert("Error al conectar con el servidor de Vórtice.");
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('placeholder-text').classList.remove('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.post("/predecir_fuga", response_model=RespuestaPrediccion, tags=["Predicción"])
async def predecir(cliente: DatosCliente):
    """
    Predice la probabilidad de fuga de un cliente.
    
    Args:
        cliente: Datos del cliente a evaluar
        
    Returns:
        RespuestaPrediccion: Probabilidad de deserción y nivel de riesgo
        
    Raises:
        HTTPException: Si hay error en el procesamiento
    """
    try:
        logger.info(f"Predicción solicitada para cliente - Edad: {cliente.edad}")
        
        # Formar el vector de características
        datos = np.array([[
            cliente.edad,
            cliente.antiguedad_meses,
            cliente.precio_membresia,
            cliente.asistencia_semanal,
            cliente.consumo_barra,
            cliente.uso_app,
            cliente.genero_masculino,
            cliente.membresia_mensual,
            cliente.membresia_trimestral
        ]])
        
        # Escalar datos
        datos_escalados = scaler.transform(datos)
        
        # Predecir probabilidades
        probabilidad = modelo.predict_proba(datos_escalados)[0][1]
        
        # Determinar nivel de riesgo
        if probabilidad > 0.7:
            nivel_riesgo = "CRÍTICO"
        elif probabilidad > 0.5:
            nivel_riesgo = "ALTO"
        elif probabilidad > 0.3:
            nivel_riesgo = "MEDIO"
        else:
            nivel_riesgo = "BAJO"
        
        alerta = bool(probabilidad > 0.50)
            
        resultado = RespuestaPrediccion(
            probabilidad_desercion=round(float(probabilidad), 4),
            alerta_de_fuga=alerta,
            nivel_riesgo=nivel_riesgo,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Predicción completada - Probabilidad: {probabilidad:.2%}, Riesgo: {nivel_riesgo}")
        return resultado
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar la predicción: {str(e)}"
        )

@app.get("/health", tags=["Salud"])
async def health_check():
    """
    Endpoint de verificación de salud del sistema.
    
    Returns:
        dict: Estado del sistema y componentes
    """
    return {
        "status": "healthy",
        "modelo_cargado": modelo is not None,
        "escalador_cargado": scaler is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/info", tags=["Información"])
async def info_modelo():
    """
    Información del modelo y sus características.
    
    Returns:
        dict: Detalles del modelo
    """
    return {
        "nombre_modelo": "Random Forest Classifier",
        "variables_entrada": [
            "edad", "antiguedad_meses", "precio_membresia",
            "asistencia_semanal", "consumo_barra", "uso_app",
            "genero_masculino", "membresia_mensual", "membresia_trimestral"
        ],
        "num_features": 9,
        "empresa": "Gimnasio Vórtice S.A.C.",
        "proyecto": "Predicción de Deserción de Clientes",
        "ciclo": "VII Ciclo Ingeniería de Sistemas UCV"
    }

@app.get("/metricas", response_model=MetricasModelo, tags=["Métricas"])
async def obtener_metricas():
    """
    Retorna las métricas de evaluación del modelo.
    
    Returns:
        MetricasModelo: Métricas del modelo
    """
    return MetricasModelo(
        accuracy=0.85,
        precision=0.82,
        recall=0.78,
        f1_score=0.80,
        auc_roc=0.88
    )
