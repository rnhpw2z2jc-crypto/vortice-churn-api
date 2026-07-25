"""
Script de Evaluación del Modelo de Machine Learning
====================================================
Evaluación completa del modelo Random Forest para predicción de deserción
de clientes del Gimnasio Vórtice S.A.C.

Incluye:
- Métricas de rendimiento (Accuracy, Precision, Recall, F1, AUC-ROC)
- Matriz de confusión
- Feature Importance
- Comparación con otros modelos
- Visualizaciones para presentación

Autor: Ingeniería de Datos - Proyecto Vórtice S.A.C.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# 1. CARGAR Y PREPARAR DATOS
# ============================================================
print("=" * 70)
print("EVALUACIÓN DEL MODELO - GIMNASIO VÓRTICE S.A.C.")
print("=" * 70)

# Cargar dataset
df = pd.read_csv("vortice_churn_data.csv")
print(f"\nDataset cargado: {len(df)} registros, {len(df.columns)} columnas")

# Preparar features (X) y target (y)
# Codificar variables categóricas
df_model = df.copy()
df_model['Genero_Num'] = (df_model['Genero'] == 'Masculino').astype(int)
df_model['Membresia_Mensual'] = (df_model['Tipo_Membresia'] == 'Mensual').astype(int)
df_model['Membresia_Trimestral'] = (df_model['Tipo_Membresia'] == 'Trimestral').astype(int)

# Seleccionar features
features = [
    'Edad', 'Antiguedad_Meses', 'Precio_Membresia_Soles',
    'Asistencia_Semanal_Promedio', 'Consumo_Barra_Soles',
    'Uso_App_Web', 'Genero_Num', 'Membresia_Mensual', 'Membresia_Trimestral'
]

X = df_model[features]
y = df_model['Fuga']

print(f"\nFeatures utilizados: {len(features)}")
print(f"Distribución del target:")
print(f"  - No Fuga (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"  - Fuga (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")

# ============================================================
# 2. DIVISIÓN TRAIN/TEST
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivisión de datos:")
print(f"  - Entrenamiento: {len(X_train)} muestras")
print(f"  - Prueba: {len(X_test)} muestras")

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN DE MÚLTIPLES MODELOS
# ============================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS")
print("=" * 70)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\nEntrenando {name}...")
    
    # Entrenar modelo
    if name == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)
    
    # Validación cruzada
    if name == 'Random Forest':
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    else:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 4. ANÁLISIS DETALLADO DEL MEJOR MODELO (RANDOM FOREST)
# ============================================================
print("\n" + "=" * 70)
print("ANÁLISIS DETALLADO - RANDOM FOREST (MEJOR MODELO)")
print("=" * 70)

rf_model = models['Random Forest']
y_pred_rf = results['Random Forest']['y_pred']
y_prob_rf = results['Random Forest']['y_prob']

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_rf)
print("\nMatriz de Confusión:")
print(f"  Verdaderos Negativos: {cm[0][0]}")
print(f"  Falsos Positivos: {cm[0][1]}")
print(f"  Falsos Negativos: {cm[1][0]}")
print(f"  Verdaderos Positivos: {cm[1][1]}")

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_rf, target_names=['No Fuga', 'Fuga']))

# Feature Importance
print("\nImportancia de Features (Feature Importance):")
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance_df.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ============================================================
# 5. GENERAR VISUALIZACIONES
# ============================================================
print("\n" + "=" * 70)
print("GENERANDO VISUALIZACIONES...")
print("=" * 70)

# Crear carpeta de salida
import os
os.makedirs('reportes', exist_ok=True)

# 5.1 Curva ROC
plt.figure(figsize=(10, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {res['auc_roc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Comparación de Modelos')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('reportes/curva_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Curva ROC guardada")

# 5.2 Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Fuga', 'Fuga'],
            yticklabels=['No Fuga', 'Fuga'])
plt.title('Matriz de Confusión - Random Forest')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.savefig('reportes/matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Matriz de Confusión guardada")

# 5.3 Feature Importance
plt.figure(figsize=(10, 6))
feature_importance_df.plot(x='Feature', y='Importance', kind='barh', 
                           legend=False, color='gold')
plt.title('Importancia de Features - Random Forest')
plt.xlabel('Importancia')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.savefig('reportes/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Feature Importance guardada")

# 5.4 Comparación de Métricas
metrics_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results],
    'AUC-ROC': [results[m]['auc_roc'] for m in results]
})

fig, ax = plt.subplots(figsize=(12, 6))
metrics_df.set_index('Modelo').plot(kind='bar', ax=ax)
plt.title('Comparación de Métricas por Modelo')
plt.ylabel('Valor')
plt.xticks(rotation=0)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('reportes/comparacion_metricas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Comparación de Métricas guardada")

# 5.5 Distribución de Probabilidades
plt.figure(figsize=(10, 6))
plt.hist(y_prob_rf[y_test == 0], bins=30, alpha=0.6, label='No Fuga', color='green')
plt.hist(y_prob_rf[y_test == 1], bins=30, alpha=0.6, label='Fuga', color='red')
plt.xlabel('Probabilidad Predicha')
plt.ylabel('Frecuencia')
plt.title('Distribución de Probabilidades - Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('reportes/distribucion_probabilidades.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Distribución de Probabilidades guardada")

# ============================================================
# 6. RESUMEN FINAL
# ============================================================
print("\n" + "=" * 70)
print("RESUMEN FINAL PARA PRESENTACIÓN")
print("=" * 70)

best_model = 'Random Forest'
print(f"\n🏆 MEJOR MODELO: {best_model}")
print(f"\n📊 MÉTRICAS PRINCIPALES:")
print(f"  • Accuracy: {results[best_model]['accuracy']:.2%}")
print(f"  • Precision: {results[best_model]['precision']:.2%}")
print(f"  • Recall: {results[best_model]['recall']:.2%}")
print(f"  • F1-Score: {results[best_model]['f1']:.2%}")
print(f"  • AUC-ROC: {results[best_model]['auc_roc']:.4f}")
print(f"  • Validación Cruzada: {results[best_model]['cv_mean']:.2%} (+/- {results[best_model]['cv_std']:.2%})")

print(f"\n📈 VARIABLES MÁS IMPORTANTES:")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}")

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • reportes/curva_roc.png")
print(f"  • reportes/matriz_confusion.png")
print(f"  • reportes/feature_importance.png")
print(f"  • reportes/comparacion_metricas.png")
print(f"  • reportes/distribucion_probabilidades.png")

print("\n" + "=" * 70)
print("EVALUACIÓN COMPLETADA EXITOSAMENTE")
print("=" * 70)
