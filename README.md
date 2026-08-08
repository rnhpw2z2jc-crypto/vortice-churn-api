# Vórtice Gym Power - API de Predicción de Deserción

## Proyecto Académico - VII Ciclo Ingeniería de Sistemas UCV

Sistema de inteligencia artificial para predicción de deserción de clientes del **Gimnasio Vórtice S.A.C.**, basado en Machine Learning y desplegado como API REST.

---

## Características Principales

- 🤖 **Modelo Random Forest** entrenado con 1,200 registros históricos
- 📊 **9 variables predictoras** del comportamiento del cliente
- 🌐 **API REST** con FastAPI y documentación Swagger
- 📱 **Interfaz web responsiva** para uso en móviles y desktop
- 🔄 **Aprendizaje continuo**: monitoreo de data drift, registro de resultados reales (feedback) y reentrenamiento del modelo
- ✅ **Cumple ISO 25010** - Estándar de calidad de software

---

## Variables de Entrada

| Variable | Descripción | Rango |
|----------|-------------|-------|
| `edad` | Edad del socio | 14-90 años |
| `antiguedad_meses` | Meses en el gimnasio | 0-120 meses |
| `precio_membresia` | Precio de la membresía | S/. 110 - 1100 |
| `asistencia_semanal` | Promedio de asistencias/semana | 0-7 |
| `consumo_barra` | Gasto en barra nutricional | S/. 0 - 150 |
| `uso_app` | Uso de plataforma web | 0 (No) / 1 (Sí) |
| `genero_masculino` | Género del socio | 0 (F) / 1 (M) |
| `membresia_mensual` | Tipo membresía mensual | 0/1 |
| `membresia_trimestral` | Tipo membresía trimestral | 0/1 |

---

## Instalación

### 1. Clonar el repositorio
```bash
cd vortice-ml-api
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Entrenar el modelo (opcional)
```bash
python evaluar_modelo.py
```

### 5. Iniciar el servidor
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

---

## Uso de la API

### Interfaz Web
Abrir en el navegador:
```
http://localhost:8000
```

### Documentación Swagger
```
http://localhost:8000/docs
```

### Documentación ReDoc
```
http://localhost:8000/redoc
```

### Ejemplo de Petición POST
```bash
curl -X POST "http://localhost:8000/predecir_fuga" \
  -H "Content-Type: application/json" \
  -d '{
    "edad": 25,
    "antiguedad_meses": 6,
    "precio_membresia": 120.0,
    "asistencia_semanal": 2.5,
    "consumo_barra": 30.0,
    "uso_app": 1,
    "genero_masculino": 1,
    "membresia_mensual": 1,
    "membresia_trimestral": 0
  }'
```

### Respuesta Esperada
```json
{
  "probabilidad_desercion": 0.4523,
  "alerta_de_fuga": false,
  "nivel_riesgo": "MEDIO",
  "timestamp": "2026-07-21T19:45:00.000000"
}
```

### Endpoints de Aprendizaje Continuo

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/drift` | GET | Estado de data drift en tiempo real (comparación contra baseline) |
| `/drift/simular` | POST | Simula la llegada de un nuevo segmento (what-if, no modifica datos) |
| `/feedback` | POST | Registra el resultado real de una predicción (`{id_prediccion, socio_se_fugo}`) |
| `/feedback/lote` | POST | Registra resultados reales en lote |
| `/feedback/estadisticas` | GET | Resumen de errores del modelo detectados (FP/FN) |
| `/retrain` | POST | Reentrena el modelo con el feedback acumulado y actualiza las métricas |

```bash
# Registrar resultado real (el modelo aprende de sus errores)
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{"id_prediccion": "PRD-20260807123456", "socio_se_fugo": true}'

# Reentrenar el modelo con el feedback acumulado
curl -X POST "http://localhost:8000/retrain"
```

### Generar feedback de demostración
```bash
python generar_feedback.py   # simula resultados reales para el historial (600 registros)
```

---

## Estructura del Proyecto

```
vortice-ml-api/
├── main.py                          # API principal FastAPI
├── evaluar_modelo.py                # Script de evaluación ML
├── generar_feedback.py              # Simulador de resultados reales (aprendizaje continuo)
├── modelo_random_forest_vortice.joblib  # Modelo entrenado
├── escalador_vortice.joblib         # Escalador StandardScaler
├── modelo_info.json                 # Metadatos del modelo (versión, drift, reentrenamientos)
├── feedback.json                    # Resultados reales registrados (aprendizaje continuo)
├── requirements.txt                 # Dependencias
├── ISO_25010.md                     # Documentación de cumplimiento ISO
├── templates/
│   └── index.html                   # Interfaz web
├── frontend/
│   └── index.html                   # Interfaz web standalone (API_BASE configurable)
├── reportes/                        # Visualizaciones generadas
│   ├── curva_roc.png
│   ├── matriz_confusion.png
│   ├── feature_importance.png
│   ├── comparacion_metricas.png
│   └── distribucion_probabilidades.png
└── README.md                        # Este archivo
```

---

## Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| Accuracy | 85.0% |
| Precision | 82.0% |
| Recall | 78.0% |
| F1-Score | 80.0% |
| AUC-ROC | 0.88 |

---

## Generación de Dataset

El dataset sintético se genera ejecutando:
```bash
cd "files (1)"
python generar_dataset_vortice.py
```

Esto creará `vortice_churn_data.csv` con 1,200 registros.

---

## Tecnologías Utilizadas

- **Backend**: FastAPI, Uvicorn
- **Machine Learning**: Scikit-learn, Random Forest
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Datos**: Pandas, NumPy
- **Serialización**: Joblib

---

## Autores

**Ingeniería de Datos - Proyecto Vórtice S.A.C.**
VII Ciclo - Escuela de Ingeniería de Sistemas
Universidad Cesar Vallejo (UCV)

---

## Licencia

Proyecto académico para fines educativos.
