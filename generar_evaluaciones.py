"""
Generador de 50 evaluaciones ficticias para el Dashboard
"""
import json
import random
from datetime import datetime, timedelta

random.seed(42)

NIVELES = {
    "BAJO": {"prob_range": (0.05, 0.29), "peso": 0.40},
    "MEDIO": {"prob_range": (0.30, 0.49), "peso": 0.25},
    "ALTO": {"prob_range": (0.50, 0.69), "peso": 0.20},
    "CRÍTICO": {"prob_range": (0.70, 0.92), "peso": 0.15},
}

RECOMENDACIONES = {
    "BAJO": "Socio estable. Invitar a programas de referidos o retos internos.",
    "MEDIO": "Enviar comunicación de bienvenida y ofrecer clase gratuita o servicio adicional.",
    "ALTO": "Ofrecer descuento del 15-20% o beneficios adicionales para retener al socio.",
    "CRÍTICO": "Acción inmediata requerida. Contactar al socio con una oferta personalizada urgente.",
}

EDADES = list(range(16, 65))
ANTIGUEDADES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 48, 60]
PRECIOS = [110.0, 120.0, 290.0, 320.0, 990.0, 1000.0, 1100.0]
ASISTENCIAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
CONSUMOS = [0, 10, 20, 30, 40, 50, 60, 75, 80, 100, 120, 150]

def generar_predicciones():
    historial = []
    ahora = datetime.now()

    for i in range(50):
        nivel = random.choices(
            list(NIVELES.keys()),
            weights=[NIVELES[n]["peso"] for n in NIVELES]
        )[0]

        prob = round(random.uniform(*NIVELES[nivel]["prob_range"]), 4)
        nivel_riesgo = nivel
        alerta = prob > 0.50
        recomendacion = RECOMENDACIONES[nivel]

        dias_atras = random.randint(0, 30)
        horas_atras = random.randint(8, 22)
        minutos_atras = random.randint(0, 59)
        timestamp = (ahora - timedelta(days=dias_atras, hours=horas_atras, minutes=minutos_atras)).isoformat()

        edad = random.choice(EDADES)
        antiguedad = random.choice(ANTIGUEDADES)
        precio = random.choice(PRECIOS)
        asistencia = random.choice(ASISTENCIAS)
        consumo = random.choice(CONSUMOS)
        uso_app = random.choice([0, 1])
        genero = random.choice([0, 1])

        if precio in [120.0]:
            m_mensual, m_trimestral = 1, 0
        elif precio in [320.0]:
            m_mensual, m_trimestral = 0, 1
        else:
            m_mensual, m_trimestral = 0, 0

        pred = {
            "id": f"PRD-{(ahora - timedelta(days=dias_atras)).strftime('%Y%m%d')}{random.randint(100000, 999999)}",
            "probabilidad_desercion": prob,
            "alerta_de_fuga": alerta,
            "nivel_riesgo": nivel_riesgo,
            "recomendacion": recomendacion,
            "timestamp": timestamp,
            "cliente_demo": {
                "edad": edad,
                "antiguedad_meses": antiguedad,
                "precio_membresia": precio,
                "asistencia_semanal": asistencia,
                "consumo_barra": consumo,
                "uso_app": uso_app,
                "genero_masculino": genero,
                "membresia_mensual": m_mensual,
                "membresia_trimestral": m_trimestral
            }
        }
        historial.append(pred)

    historial.sort(key=lambda x: x["timestamp"], reverse=True)

    with open("historial_predicciones.json", "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    print(f"{len(historial)} predicciones generadas")
    
    total = len(historial)
    en_riesgo = sum(1 for p in historial if p["alerta_de_fuga"])
    seguros = total - en_riesgo
    ahorro = seguros * 1800
    
    print(f"\nEstadisticas:")
    print(f"   Total: {total}")
    print(f"   En riesgo: {en_riesgo} ({en_riesgo/total*100:.0f}%)")
    print(f"   Seguros: {seguros} ({seguros/total*100:.0f}%)")
    print(f"   Ahorro estimado: S/. {ahorro:,}")

    from collections import Counter
    conteo = Counter(p["nivel_riesgo"] for p in historial)
    print(f"\nDistribucion:")
    for nivel in ["BAJO", "MEDIO", "ALTO", "CRÍTICO"]:
        n = conteo.get(nivel, 0)
        barra = "#" * n
        print(f"   {nivel:>8}: {n:>2} {barra}")

if __name__ == "__main__":
    generar_predicciones()
