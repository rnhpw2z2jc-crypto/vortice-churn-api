"""
Simulador de Feedback Real (resultados observados)
==================================================
Genera los resultados reales (quien realmente se fugo) para las predicciones
del historial. El modelo aprende de estos resultados mediante reentrenamiento.

Los resultados reales se simulan con una regla de negocio aprendible a partir
de las features, de modo que el Random Forest pueda detectarla en el
reentrenamiento (aprendizaje continuo / correccion de data drift).
"""
import json
import random
from datetime import datetime
import numpy as np

random.seed(7)

FEEDBACK_FILE = "feedback.json"
HISTORY_FILE = "historial_predicciones.json"


def prob_fuga_real(cliente):
    """Regla de negocio: perfil de socio con mayor probabilidad de fuga."""
    riesgo = 0
    if cliente["asistencia_semanal"] < 3.0:
        riesgo += 1
    if cliente["uso_app"] == 0:
        riesgo += 1
    if cliente["consumo_barra"] < 50:
        riesgo += 1
    if cliente["precio_membresia"] >= 900:
        riesgo += 1
    if cliente["precio_membresia"] <= 120 and cliente["antiguedad_meses"] > 24:
        riesgo += 1
    if cliente["antiguedad_meses"] <= 6:
        riesgo += 1
    ruido = random.uniform(-0.8, 0.8)
    return 1.0 / (1.0 + np.exp(-(riesgo - 2.6 + ruido)))


def generar_feedback():
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        historial = json.load(f)

    feedback = []
    desacuerdos = 0
    para_confirmar = 0
    for p in historial:
        if not p.get("cliente_demo"):
            continue
        prob_real = float(prob_fuga_real(p["cliente_demo"]))
        socio_se_fugo = bool(prob_real > 0.5)
        predijo_fuga = bool(p.get("alerta_de_fuga"))
        if predijo_fuga != socio_se_fugo:
            desacuerdos += 1
        if predijo_fuga and socio_se_fugo:
            para_confirmar += 1
        feedback.append({
            "id_prediccion": p["id"],
            "probabilidad_predicha": p.get("probabilidad_desercion"),
            "socio_se_fugo": socio_se_fugo,
            "fecha_prediccion": p.get("timestamp"),
            "fecha_feedback": datetime.now().isoformat(),
            "cliente_demo": p["cliente_demo"]
        })

    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    total = len(feedback)
    fugas = sum(1 for fb in feedback if fb["socio_se_fugo"])
    seguros = total - fugas
    print(f"{total} resultados reales generados")
    print(f"   Se fugaron:    {fugas} ({fugas/total*100:.0f}%)")
    print(f"   Se quedaron:   {seguros} ({seguros/total*100:.0f}%)")
    print(f"   Modelo coincidio con la realidad: {total - desacuerdos} ({100 - desacuerdos/total*100:.0f}%)")
    print(f"   Errores del modelo (aprendera de ellos): {desacuerdos}")
    print(f"   Fugas confirmadas por el modelo: {para_confirmar}")


if __name__ == "__main__":
    generar_feedback()
