import pandas as pd
import numpy as np
import pickle

# Para visualización (matrices de confusión, etc.)
import matplotlib.pyplot as plt

# Para preprocesamiento y modelos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Para métricas y evaluación
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

data = pd.read_csv("C:/Users/cristian.gonzalez/Downloads/MLOps_Heroku/MLOps_Obesity/ObesityDataSet_raw_and_data_sinthetic.csv")

print("Primeras filas del dataset:")
print(data.head())

print("\nInformación del dataset:")
print(data.info())

# La columna objetivo (clase a predecir) se llama 'NObeyesdad'
# (Ejemplo: Insufficient_Weight, Normal_Weight, Overweight, Obesity, etc.)
target_col = 'NObeyesdad'

# Identificamos características y variable objetivo
X = data.drop(columns=[target_col])
y = data[target_col]

# Algunas variables del dataset podrían ser categóricas (ej. 'Gender', 'family_history', etc.)
# Necesitamos transformarlas a numéricas. Aquí hacemos un LabelEncoder básico
categorical_cols = X.select_dtypes(include='object').columns

if len(categorical_cols) > 0:
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

y_le = LabelEncoder()
y = y_le.fit_transform(y)

# -----------------------------------------------------------
# Dividir datos en entrenamiento y prueba
# -----------------------------------------------------------
# Usamos estratificación para respetar la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -----------------------------------------------------------
# Escalado de características
# -----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------
# Entrenar modelo de Regresión Logística (multiclase)
# -----------------------------------------------------------
# Para problemas multiclase, podemos usar multi_class='ovr' (one-vs-rest)
lr = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    multi_class='ovr'
)
lr.fit(X_train_scaled, y_train)

# Predicciones
y_pred_lr = lr.predict(X_test_scaled)

# Probabilidades (para métricas como AUC en multiclase con roc_auc_score)
# roc_auc_score en multiclase se define con multi_class='ovr' o 'ovo'
y_prob_lr = lr.predict_proba(X_test_scaled)

# Métricas de evaluación
print("\n--- Regresión Logística (Multiclase) ---")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_lr))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_lr))

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Exactitud (Accuracy): {accuracy_lr:.4f}")

try:
    roc_auc_lr = roc_auc_score(y_test, y_prob_lr, multi_class='ovr')
    print(f"ROC AUC (one-vs-rest): {roc_auc_lr:.4f}")
except ValueError:
    print("No se pudo calcular el ROC AUC para multiclase (verifica versión de scikit-learn)")

# -----------------------------------------------------------
# Entrenar modelo de Bosque Aleatorio (multiclase)
# -----------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Predicciones
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)

# Métricas
print("\n--- Bosque Aleatorio (Multiclase) ---")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rf))

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Exactitud (Accuracy): {accuracy_rf:.4f}")

try:
    roc_auc_rf = roc_auc_score(y_test, y_prob_rf, multi_class='ovr')
    print(f"ROC AUC (one-vs-rest): {roc_auc_rf:.4f}")
except ValueError:
    print("No se pudo calcular el ROC AUC para multiclase (verifica versión de scikit-learn)")

# -----------------------------------------------------------
# Guardar los modelos y el scaler (opcional)
# -----------------------------------------------------------
with open('modelo_regresion_logistica.pkl', 'wb') as f:
    pickle.dump(lr, f)

with open('modelo_bosque_aleatorio.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModelos y scaler guardados en archivos .pkl")
