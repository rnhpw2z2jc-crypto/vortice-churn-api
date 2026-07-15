from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Gimnasio Vórtice S.A.C. - API de IA")

# Cargar el modelo y el escalador
modelo = joblib.load('modelo_random_forest_vortice.joblib')
scaler = joblib.load('escalador_vortice.joblib')

# Clase de datos de entrada (Debe coincidir exactamente con lo que envía el JavaScript)
class DatosCliente(BaseModel):
    edad: float
    antiguedad_meses: float
    precio_membresia: float
    asistencia_semanal: float
    consumo_barra: float
    uso_app: int
    genero_masculino: int
    membresia_mensual: int
    membresia_trimestral: int

@app.post("/predecir_fuga")
def predecir(cliente: DatosCliente):
    try:
        # Formar el vector de entrada para el modelo
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
        
        # Determinar el nivel de riesgo en base a umbrales de negocio
        if probabilidad <= 0.40:
            riesgo = "BAJO"
        elif probabilidad <= 0.70:
            riesgo = "MODERADO"
        else:
            riesgo = "ALTO"
            
        return {
            "probabilidad_desercion": round(float(probabilidad), 4),
            "nivel_riesgo": riesgo
        }
    except Exception as e:
        # Si algo falla en Python, esto evitará que la API muera en silencio
        return {"error": str(e), "detalle": "Error interno al procesar los vectores."}

@app.get("/", response_class=HTMLResponse)
def home():
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
                <span class="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-[10px] sm:text-xs px-2.5 py-1 rounded-full flex items-center">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5 animate-ping"></span> Servidor Activo
                </span>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-4xl mx-auto p-4 sm:p-6 w-full flex-grow flex flex-col justify-center">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
                
                <!-- Formulario -->
                <div class="bg-[#141414] p-5 sm:p-6 rounded-2xl border border-zinc-800/60 shadow-2xl flex flex-col justify-between">
                    <div>
                        <h2 class="text-base sm:text-lg font-bold text-zinc-100 mb-4 flex items-center border-b border-zinc-800/60 pb-2">
                            <i class="fa-solid fa-user-gear mr-2 text-gold-500"></i> Datos del Socio a Evaluar
                        </h2>
                        
                        <form id="form-predict" class="space-y-3.5">
                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Edad del Socio</label>
                                    <input type="number" min="14" max="90" id="edad" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all" placeholder="Ej: 26" required>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Asistencia Semanal</label>
                                    <input type="number" step="0.1" min="0" max="7" id="asistencia" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all" placeholder="Ej: 3.5" required>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Antigüedad (Meses)</label>
                                    <input type="number" min="0" id="antiguedad" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all" placeholder="Ej: 6" required>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Consumo Barra (S/.)</label>
                                    <input type="number" step="0.1" min="0" id="consumo" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all" placeholder="Ej: 45.90" required>
                                </div>
                            </div>

                            <div class="grid grid-cols-2 gap-3.5">
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Género</label>
                                    <select id="genero" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all">
                                        <option value="1">Masculino</option>
                                        <option value="0">Femenino</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Plataforma Web</label>
                                    <select id="uso_app" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all">
                                        <option value="1">Sí usa la Web</option>
                                        <option value="0">No usa la Web</option>
                                    </select>
                                </div>
                            </div>

                            <div>
                                <label class="block text-[11px] font-semibold text-zinc-400 uppercase tracking-wider mb-1">Tipo de Membresía</label>
                                <select id="membresia" class="w-full bg-[#1c1c1c] border border-zinc-800 rounded-lg p-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-gold-500 transition-all">
                                    <option value="mensual">Membresía Mensual (S/. 120.00)</option>
                                    <option value="trimestral">Membresía Trimestral (S/. 320.00)</option>
                                    <option value="anual">Membresía Anual (S/. 1100.00)</option>
                                </select>
                            </div>

                            <button type="submit" class="w-full mt-4 bg-gradient-to-r from-gold-500 to-gold-600 hover:from-gold-400 hover:to-gold-500 text-black active:scale-[0.98] transition-all py-3 rounded-xl font-bold text-sm tracking-wider flex justify-center items-center shadow-lg shadow-gold-500/10">
                                <i class="fa-solid fa-brain mr-2"></i> Calcular Riesgo de Fuga
                            </button>
                        </form>
                    </div>
                </div>

                <!-- Resultados -->
                <div class="bg-[#141414] p-6 rounded-2xl border border-zinc-800/60 shadow-2xl min-h-[380px] flex flex-col justify-center items-center text-center relative overflow-hidden" id="card-resultado">
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
                    </div>
                </div>

            </div>
        </main>

        <!-- Footer -->
        <footer class="bg-[#0e0e0e] p-4 border-t border-zinc-900 text-center text-[10px] sm:text-xs text-zinc-600">
            © 2026 VÓRTICE GYM POWER - VII Ciclo Escuela de Ingeniería de Sistemas UCV. Todos los derechos reservados.
        </footer>

        <!-- Scripts -->
        <script>
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

                // PAYLOAD EXACTO (Los nombres de las llaves deben coincidir perfectamente con la clase DatosCliente de Python)
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
                    
                    if (data.error) {
                        console.error("Error devuelto por la API:", data.error);
                        alert("Error en el modelo: " + data.error);
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

                    if (data.nivel_riesgo === "ALTO") {
                        badge.className = "px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest bg-rose-500/10 text-rose-400 border border-rose-500/20 uppercase";
                        badge.innerText = "ALERTA: ALTO RIESGO DE FUGA";
                        
                        document.getElementById('probabilidad-valor').className = "text-5xl sm:text-6xl font-black text-rose-500";
                        titulo.innerText = "Socio Propenso a Retirarse";
                        titulo.className = "text-base sm:text-lg font-black text-rose-400";
                        desc.innerText = "El algoritmo identificó un patrón crítico de inactividad prolongada y nulo consumo. La deserción es inminente.";
                        rec.innerText = "Contacto inmediato vía llamada telefónica por el administrador. Ofrecer reajuste de congelamiento de membresía gratuito o un descuento drástico por renovación.";
                    
                    } else if (data.nivel_riesgo === "MODERADO") {
                        badge.className = "px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest bg-amber-500/10 text-amber-400 border border-amber-500/20 uppercase";
                        badge.innerText = "OBSERVACIÓN: RIESGO MODERADO";
                        
                        document.getElementById('probabilidad-valor').className = "text-5xl sm:text-6xl font-black text-amber-500";
                        titulo.innerText = "Socio en Alerta Preventiva";
                        titulo.className = "text-base sm:text-lg font-black text-amber-400";
                        desc.innerText = "Se observa una reducción gradual en la asistencia semanal y una falta de uso de la plataforma web. Patrón de pérdida de hábito.";
                        rec.innerText = "Enviar mensaje de WhatsApp con rutina motivacional. Ofrecer un pase libre para un invitado de fin de semana o un batido gratis en la barra de suplementos.";
                    
                    } else {
                        badge.className = "px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 uppercase";
                        badge.innerText = "SOCIO ACTIVO Y CONFORME";
                        
                        document.getElementById('probabilidad-valor').className = "text-5xl sm:text-6xl font-black text-emerald-500";
                        titulo.innerText = "Cliente Saludable";
                        titulo.className = "text-base sm:text-lg font-black text-emerald-400";
                        desc.innerText = "El socio mantiene una asistencia constante alineada a sus promedios históricos y consume activamente productos de valor agregado.";
                        rec.innerText = "Mantener excelente estándar de servicio en sala de musculación. Invitar a participar en retos internos del gimnasio o felicitación automatizada por hitos de asistencia.";
                    }

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