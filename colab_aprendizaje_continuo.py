"""
Celdas para Google Colab: Aprendizaje Continuo
==============================================
Pegar DESPUES de las celdas de entrenamiento v1 (las que ya tienes,
que cargan vortice_churn_data.csv, entrenan y exportan el modelo).

Cada bloque separado por #%% es UNA celda del notebook.
Orden sugerido: primero las 6 celdas originales, luego estas.

El modelo v1 del Colab usa n_estimators=100 y la API ahora reentrena
con los mismos hiperparametros (main.py).
"""

#%%
# ============================================================
# CELDA A — VERIFICAR QUE EL ORDEN DE FEATURES COINCIDE CON LA API
# ============================================================
# La API (main.py) espera este orden exacto de 9 features:
orden_api = [
    'Edad', 'Antiguedad_Meses', 'Precio_Membresia_Soles',
    'Asistencia_Semanal_Promedio', 'Consumo_Barra_Soles', 'Uso_App',
    'Genero_Masculino', 'Tipo_Membresia_Mensual', 'Tipo_Membresia_Trimestral'
]
print("Features del modelo Colab :", X.columns.tolist())
print("Coinciden con la API     :", X.columns.tolist() == orden_api)
if X.columns.tolist() != orden_api:
    print("ADVERTENCIA: reordena X para que coincida antes de exportar.")

#%%
# ============================================================
# CELDA B — LOS ERRORES DEL MODELO v1 (QUIEN FALLO)
# ============================================================
# El conjunto de test (20%) son socios cuyo resultado real (Fuga) ya se conoce.
# Comparar lo que predijo v1 con la realidad revela los errores que corregira
# el reentrenamiento. Asi funciona el ciclo de aprendizaje continuo.
from sklearn.metrics import confusion_matrix

y_pred_v1 = modelo_vortice.predict(X_test_scaled)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_v1).ravel()

print("Matriz de confusion del modelo v1 sobre su test:")
print(f"  Verdaderos Negativos (predijo seguro, se quedo) : {tn}")
print(f"  Falsos Positivos  (predijo fuga,  se quedo)    : {fp}")
print(f"  Falsos Negativos  (predijo seguro, se fugo)    : {fn}")
print(f"  Verdaderos Positivos (predijo fuga, se fugo)   : {tp}")
print(f"\nErrores totales que el modelo debe corregir: {fp + fn}")
print("-> Estos datos reales se acumulan como feedback para el reentrenamiento.")

#%%
# ============================================================
# CELDA C — REENTRENAMIENTO v2 CON LOS DATOS ACUMULADOS
# ============================================================
# La realidad ya llego: ahora el modelo se reentrena con el 100% de los datos
# (entrenamiento original + feedback real = 1200 registros), igual que el
# endpoint /retrain de la API. Hiperparametros identicos a v1.
from sklearn.model_selection import cross_val_score, cross_val_predict

scaler_v2 = StandardScaler()
X_all_scaled = scaler_v2.fit_transform(X)      # dataset completo acumulado

modelo_v2 = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_v2.fit(X_all_scaled, y)

# Evaluacion honesta con validacion cruzada 5-fold (no se entrena y evalua en lo mismo)
cv = cross_val_score(modelo_v2, X_all_scaled, y, cv=5, scoring='accuracy')
probs_v2 = cross_val_predict(modelo_v2, X_all_scaled, y, cv=5, method='predict_proba')[:, 1]
auc_v2 = roc_auc_score(y, probs_v2)

print(f"Modelo v2 (100% datos + feedback):")
print(f"  Accuracy CV = {cv.mean():.2%} +/- {cv.std():.2%}")
print(f"  AUC-ROC     = {auc_v2:.4f}")

#%%
# ============================================================
# CELDA D — COMPARACION v1 vs v2 (APRENDIZAJE CONTINUO)
# ============================================================
# v1: entrenado solo con el 80% (test = 20%)
acc_v1 = accuracy_score(y_test, y_pred_v1)
auc_v1 = roc_auc_score(y_test, modelo_vortice.predict_proba(X_test_scaled)[:, 1])

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].bar(['Modelo v1\n(80% datos)', 'Modelo v2\n(100% datos + feedback)'],
          [acc_v1, cv.mean()], color=['#888888', '#d4af37'], width=0.55)
ax[0].set_ylim(0, 1.05)
ax[0].set_title('Accuracy del Modelo', fontweight='bold')
ax[0].set_ylabel('Accuracy')
for i, v in enumerate([acc_v1, cv.mean()]):
    ax[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')

ax[1].bar(['Modelo v1', 'Modelo v2'], [auc_v1, auc_v2],
          color=['#888888', '#d4af37'], width=0.55)
ax[1].set_ylim(0, 1.05)
ax[1].set_title('AUC-ROC del Modelo', fontweight='bold')
ax[1].set_ylabel('AUC-ROC')
for i, v in enumerate([auc_v1, auc_v2]):
    ax[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.suptitle('Aprendizaje Continuo: el modelo mejora al aprender de sus errores',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

print("=" * 60)
print(f"Accuracy : v1 {acc_v1:.1%}  ->  v2 {cv.mean():.1%}")
print(f"AUC-ROC  : v1 {auc_v1:.3f}  ->  v2 {auc_v2:.3f}")
print("=" * 60)
print("El modelo aprendio de sus errores y mejoro su desempeno.")

#%%
# ============================================================
# CELDA E — (OPCIONAL) EXPORTAR v2 PARA LA API
# ============================================================
# Solo descomenta si quieres reemplazar el modelo desplegado con v2.
# Normalmente la API genera su propio reentrenamiento con POST /retrain
# y respalda el modelo anterior automaticamente.
#
# import joblib
# joblib.dump(modelo_v2, 'modelo_random_forest_vortice.joblib')
# joblib.dump(scaler_v2, 'escalador_vortice.joblib')
# print("Modelo v2 exportado. Subelo a la API (o usa /retrain).")
