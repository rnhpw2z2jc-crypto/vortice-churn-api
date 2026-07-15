from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Gimnasio Vórtice S.A.C. - API de IA")

# Cargar el modelo y el escalador
modelo = joblib.load('modelo_random_forest_vortice.joblib')
scaler = joblib.load('escalador_vortice.joblib')

# Estructura de datos para la API
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
    datos = np.array([[
        cliente.asistencia_semanal,
        cliente.antiguedad_meses,
        cliente.consumo_barra,
        cliente.uso_app,
        cliente.genero_masculino,
        cliente.membresia_trimestral,
        cliente.membresia_anual
    ]])
    
    datos_escalados = scaler.transform(datos)
    probabilidad = modelo.predict_proba(datos_escalados)[0][1]
    alerta = bool(probabilidad > 0.50) # Umbral óptimo de decisión
    
    return {
        "probabilidad_desercion": round(float(probabilidad), 4),
        "alerta_de_fuga": alerta
    }

# Ruta de inicio que sirve la interfaz web interactiva
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Vórtice S.A.C. - IA Predicción de Deserción</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body class="bg-slate-900 text-white min-h-screen flex flex-col justify-between font-sans">
        <header class="bg-slate-950 border-b border-slate-800 p-4 shadow-lg">
            <div class="max-w-6xl mx-auto flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <div class="bg-indigo-600 p-2.5 rounded-lg text-white">
                        <i class="fa-solid fa-dumbbell text-xl animate-pulse"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold tracking-wider text-indigo-400">GIMNASIO VÓRTICE S.A.C.</h1>
                        <p class="text-xs text-slate-400">Plataforma de Inteligencia Artificial Predictiva</p>
                    </div>
                </div>
                <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs px-2.5 py-1 rounded-full flex items-center">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-ping"></span> Servidor Activo
                </span>
            </div>
        </header>

        <main class="max-w-4xl mx-auto p-6 w-full flex-grow flex flex-col justify-center">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                
                <div class="bg-slate-950/50 p-6 rounded-2xl border border-slate-800 shadow-xl">
                    <h2 class="text-lg font-semibold text-slate-200 mb-4 flex items-center">
                        <i class="fa-solid fa-user-gear mr-2 text-indigo-500"></i> Datos del Socio a Evaluar
                    </h2>
                    
                    <form id="form-predict" class="space-y-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Asistencias Semanales</label>
                            <input type="number" step="0.1" min="0" max="7" id="asistencia" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Ej: 3.5" required>
                        </div>

                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Antigüedad (Meses)</label>
                                <input type="number" min="0" id="antiguedad" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Ej: 6" required>
                            </div>
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Consumo Barra (S/.)</label>
                                <input type="number" step="0.1" min="0" id="consumo" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Ej: 45.90" required>
                            </div>
                        </div>

                        <div class="grid grid-cols-2 gap-4">
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Género</label>
                                <select id="genero" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                    <option value="1">Masculino</option>
                                    <option value="0">Femenino</option>
                                </select>
                            </div>
                            <div>
                                <label class="block text-xs font-semibold text-slate-400 mb-1">Usa App del Gimnasio</label>
                                <select id="uso_app" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                    <option value="1">Sí, usa la web</option>
                                    <option value="0">No usa la web</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Tipo de Membresía</label>
                            <select id="membresia" class="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                <option value="mensual">Membresía Mensual</option>
                                <option value="trimestral">Membresía Trimestral</option>
                                <option value="anual">Membresía Anual</option>
                            </select>
                        </div>

                        <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-500 active:scale-95 transition-all py-3 rounded-lg font-bold text-sm tracking-wider flex justify-center items-center">
                            <i class="fa-solid fa-brain mr-2"></i> Calcular Riesgo de Fuga
                        </button>
                    </form>
                </div>

                <div class="bg-slate-950/50 p-6 rounded-2xl border border-slate-800 shadow-xl min-h-[400px] flex flex-col justify-center items-center text-center relative overflow-hidden" id="card-resultado">
                    <div id="loading" class="hidden flex-col items-center space-y-3">
                        <div class="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
                        <p class="text-sm text-slate-400">Analizando patrones de comportamiento...</p>
                    </div>

                    <div id="placeholder-text" class="flex flex-col items-center space-y-4">
                        <i class="fa-solid fa-chart-line text-5xl text-slate-600"></i>
                        <p class="text-slate-400 text-sm max-w-[280px]">Ingresa los datos de comportamiento del socio en la izquierda para predecir si continuará o se fugará.</p>
                    </div>

                    <div id="resultado-content" class="hidden w-full flex flex-col items-center space-y-4">
                        <span id="alerta-badge" class="px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider"></span>
                        
                        <div class="relative flex items-center justify-center">
                            <div class="text-5xl font-black" id="probabilidad-valor">0%</div>
                        </div>
                        
                        <h3 class="text-lg font-bold" id="resultado-titulo"></h3>
                        <p class="text-slate-400 text-sm max-w-[300px]" id="resultado-descripcion"></p>
                        
                        <div class="w-full bg-slate-900 border border-slate-800 p-4 rounded-xl text-left mt-2">
                            <span class="text-xs text-indigo-400 font-semibold block mb-1">Plan de acción recomendado:</span>
                            <p class="text-xs text-slate-300" id="recomendacion-texto"></p>
                        </div>
                    </div>
                </div>

            </div>
        </main>

        <footer class="bg-slate-950 p-4 border-t border-slate-800 text-center text-xs text-slate-500">
            © 2026 Gimnasio Vórtice S.A.C. - VII Ciclo Escuela de Ingeniería de Sistemas UCV. Todos los derechos reservados.
        </footer>

        <script>
            document.getElementById('form-predict').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                // Mostrar cargador
                document.getElementById('placeholder-text').classList.add('hidden');
                document.getElementById('resultado-content').classList.add('hidden');
                document.getElementById('loading').classList.remove('hidden');

                // Obtener datos del formulario
                const asistencia_semanal = parseFloat(document.getElementById('asistencia').value);
                const count_meses = parseFloat(document.getElementById('antiguedad').value);
                const consumo = parseFloat(document.getElementById('consumo').value);
                const genero = parseInt(document.getElementById('genero').value);
                const uso_app = parseInt(document.getElementById('uso_app').value);
                const membresia = document.getElementById('membresia').value;

                // Mapear membresía a las variables dummy correspondientes
                let m_trimestral = 0;
                let m_anual = 0;
                if (membresia === "trimestral") m_trimestral = 1;
                if (membresia === "anual") m_anual = 1;

                const payload = {
                    asistencia_semanal: asistencia_semanal,
                    antiguedad_meses: count_meses,
                    consumo_barra: consumo,
                    uso_app: uso_app,
                    genero_masculino: genero,
                    membresia_trimestral: m_trimestral,
                    membresia_anual: m_anual
                };

                try {
                    const response = await fetch('/predecir_fuga', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    const data = await response.json();
                    
                    // Ocultar cargador
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('resultado-content').classList.remove('hidden');

                    const prob_pct = (data.probabilidad_desercion * 100).toFixed(1);
                    document.getElementById('probabilidad-valor').innerText = prob_pct + "%";

                    const badge = document.getElementById('alerta-badge');
                    const titulo = document.getElementById('resultado-titulo');
                    const desc = document.getElementById('resultado-descripcion');
                    const rec = document.getElementById('recomendacion-texto');

                    if (data.alerta_de_fuga) {
                        badge.className = "px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider bg-rose-500/10 text-rose-400 border border-rose-500/20";
                        badge.innerText = "RIESGO DE DESERCIÓN DETECTADO";
                        
                        document.getElementById('probabilidad-valor').className = "text-5xl font-black text-rose-500";
                        titulo.innerText = "¡Socio Propenso a Retirarse!";
                        titulo.className = "text-lg font-bold text-rose-400";
                        desc.innerText = "El algoritmo de Inteligencia Artificial detectó que el socio tiene un patrón conductual de alta inestabilidad.";
                        rec.innerText = "Ejecutar alerta inmediata de fidelización. Ofrecer pase libre de barra nutricional o promoción en renovación anual para reengancharlo.";
                    } else {
                        badge.className = "px-4 py-1.5 rounded-full text-xs font-semibold tracking-wider bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
                        badge.innerText = "CLIENTE ESTABLE";
                        
                        document.getElementById('probabilidad-valor').className = "text-5xl font-black text-emerald-500";
                        titulo.innerText = "Socio Conforme y Activo";
                        titulo.className = "text-lg font-bold text-emerald-400";
                        desc.innerText = "El cliente posee hábitos estables dentro de las instalaciones y no muestra intenciones de cancelar.";
                        rec.innerText = "Mantener estándar de servicio. Se sugiere enviarle encuestas de satisfacción periódicas automáticas vía correo electrónico.";
                    }

                } catch (error) {
                    console.error("Error al conectar con el servidor API:", error);
                    alert("Error al procesar la predicción en el servidor.");
                    document.getElementById('loading').classList.add('hidden');
                    document.getElementById('placeholder-text').classList.remove('hidden');
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)