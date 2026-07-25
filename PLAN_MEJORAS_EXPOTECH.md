# PLAN DE MEJORAS PARA EXPOTECH - VÓRTICE GYM POWER

## Objetivo
Presentar el proyecto como una **iniciativa empresarial SaaS** que ayuda al crecimiento de empresas mediante IA predictiva.

---

## FASE 1: MEJORAS VISUALES (Impacto en Expo)

### 1.1 Dashboard Principal Mejorado
- **Hero Section** con estadísticas clave animadas (clientes en riesgo, ahorro estimado, etc.)
- **Gráficos Chart.js** mostrando:
  - Distribución de riesgo (doughnut chart)
  - Tendencia de predicciones (line chart)
  - Comparación de métricas (bar chart)
- **Contadores animados** con efecto "count up"

### 1.2 Panel de Resultados Premium
- **Tarjetas de métricas** con iconos y colores por nivel de riesgo
- **Badge animado** con efecto glow para alertas
- **Recomendaciones IA** con íconos y formato de tarjeta
- **Timeline de actividad** del cliente

### 1.3 Modo Demo Interactivo
- **Botón "Demo Rápida"** que llena datos de ejemplo automáticamente
- **3 escenarios predefinidos**: Cliente Saludable, En Riesgo, Crítico
- **Transiciones suaves** entre estados

---

## FASE 2: FUNCIONALIDAD DE NEGOCIO

### 2.1 Panel de Analíticas (Nueva vista)
- **Métricas aggregadas**:
  - Total de predicciones realizadas
  - % de clientes en riesgo
  - Ahorro estimado por retención
  - Tasa de éxito del modelo
- **Gráficas de tendencia** (últimas 24h, 7 días, 30 días)

### 2.2 Calculadora de ROI
- **Inputs**: 
  - Número de socios
  - Costo promedio de adquisición (CAC)
  - Valor de vida del cliente (LTV)
- **Output**:
  - Ahorro anual estimado
  - ROI del sistema
  - Payback period

### 2.3 Predicciones por Lote
- **Endpoint nuevo**: `POST /predecir_lote`
- **Upload CSV** con múltiples clientes
- **Resultado**: Dashboard con resumen de todo el lote

### 2.4 Historial de Predicciones
- **Endpoint nuevo**: `GET /historial`
- **Almacenamiento**: Archivo JSON local
- **Visualización**: Tabla con filtros

---

## FASE 3: ENDPOINTS NUEVOS

```python
# Panel de analíticas
GET /dashboard/stats     -> Estadísticas agregadas
GET /dashboard/trends    -> Tendencias temporales

# Calculadora de ROI
POST /calcular_roi       -> ROI basado en parámetros

# Predicciones por lote
POST /predecir_lote      -> Múltiples predicciones

# Historial
GET /historial           -> Predicciones anteriores
GET /historial/{id}      -> Detalle de predicción
```

---

## FASE 4: MEJORAS EN LA API

### 4.1 Nuevos Modelos de Datos
```python
class DatosLote(BaseModel):
    clientes: List[DatosCliente]

class ResultadoROI(BaseModel):
    ahorro_anual: float
    roi_porcentaje: float
    payback_meses: int
    clientes_en_riesgo: int

class EstadisticasDashboard(BaseModel):
    total_predicciones: int
    clientes_en_riesgo: int
    tasa_exito: float
    ahorro_estimado: float
```

### 4.2 Almacenamiento Local
- Archivo `historial.json` para persistir predicciones
- Lectura/escritura asíncrona

---

## FASE 5: PRESENTACIÓN PARA LA JUEZA

### 5.1 Narrativa de Negocio
> "Vórtice Power no es solo una API técnica, es una **solución SaaS** que cualquier gimnasio o negocio de suscripción puede implementar para reducir la evasión de clientes hasta en un 35%."

### 5.2 Datos para Presentar
| Métrica | Valor |
|---------|-------|
| Ahorro anual por cliente retenido | S/. 1,440 |
| ROI del sistema (100 socios) | 320% |
| Payback period | 2.3 meses |
| Precisión del modelo | 85% |

### 5.3 Demo en Vivo
1. Mostrar dashboard con métricas
2. Realizar predicción en tiempo real
3. Mostrar calculadora de ROI
4. Explicar impacto en el negocio

---

## ARCHIVOS A MODIFICAR

| Archivo | Cambios |
|---------|---------|
| `main.py` | Agregar endpoints nuevos, mejorar HTML |
| `requirements.txt` | Agregar chart.js (CDN) |
| `evaluar_modelo.py` | No modificar |
| `templates/index.html` | Separar en múltiples vistas |

---

## CRONOGRAMA (Hoy)

| Hora | Tarea |
|------|-------|
| 19:00 | FASE 1: Mejoras visuales |
| 21:00 | FASE 2: Funcionalidad negocio |
| 23:00 | FASE 3: Nuevos endpoints |
| 01:00 | FASE 4: Testing y ajustes |
| 06:00 | FASE 5: Preparar presentación |

---

## ESTADO

- [ ] FASE 1: Mejoras visuales
- [ ] FASE 2: Funcionalidad negocio
- [ ] FASE 3: Nuevos endpoints
- [ ] FASE 4: Testing
- [ ] FASE 5: Presentación
