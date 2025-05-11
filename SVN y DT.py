# ============================================
# Clasificadores Clásicos: Árbol de Decisión y SVM con Kernel Gaussiano
# ============================================
"""
Este script extiende la Parte 2 del clasificador neuronal, manteniendo la lógica de:
- Carga y partición de datos
- Preprocesamiento: SMOTETomek, QuantileTransformer, PCA
- Búsqueda de umbral óptimo basándose en F1-score
- Evaluación con métricas y gráficas
"""

# 1. Importaciones
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    PrecisionRecallDisplay, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

from imblearn.combine import SMOTETomek

# 2. Configuración Inicial y Datos
np.random.seed(42)
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Preprocesamiento y PCA (reutilizable)
def entrenar_pca(X_train_scaled, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(X_train_scaled)
    return pca

# Función para entrenar y validar clasificadores clásicos
def entrenar_modelo_clasico(config, X_train_full, y_train_full, variance_threshold=0.95, tipo='tree'):
    # Partición entrenamiento/validación
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    # Balanceo con SMOTETomek
    sm = SMOTETomek(sampling_strategy=1, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    # Escalado
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_res_scaled = scaler.fit_transform(X_res)
    X_val_scaled = scaler.transform(X_val)
    # PCA
    pca = entrenar_pca(X_res_scaled, variance_threshold)
    X_res_pca = pca.transform(X_res_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    # Selección de modelo
    if tipo == 'tree':
        modelo = DecisionTreeClassifier(
            max_depth=config.get('max_depth', None),
            min_samples_leaf=config.get('min_samples_leaf', 1),
            random_state=42
        )
    elif tipo == 'svm':
        modelo = SVC(
            kernel='rbf', C=config.get('C', 1.0),
            gamma=config.get('gamma', 'scale'), probability=True, random_state=42
        )
    else:
        raise ValueError("Tipo de modelo no soportado: use 'tree' o 'svm'")
    # Entrenamiento
    modelo.fit(X_res_pca, y_res)
    # Predicción probabilística en validación
    y_val_prob = modelo.predict_proba(X_val_pca)[:, 1]
    # Búsqueda de umbral óptimo (F1)
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    umbral_opt = thresholds[np.nanargmax(f1_scores)]
    return modelo, scaler, pca, umbral_opt

# Función de evaluación común
def evaluar_modelo_clasico(modelo, scaler, pca, X_test, y_test, umbral, nombre='Modelo'):
    # Preprocesar test
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    # Predicción
    y_prob = modelo.predict_proba(X_test_pca)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)
    # Métricas
    auc_pr = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{nombre} - AUC-PR: {auc_pr:.4f}")
    print(f"{nombre} - Precisión: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    # Curva Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"{nombre}: Curva Precisión-Recall (AUC-PR = {auc_pr:.4f})")
    plt.show()
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=['No failed','Failed'], yticklabels=['No failed','Failed'])
    plt.title(f"{nombre}: Matriz de Confusión")
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    return y_pred

# 4. Ejecución Principal de Clasificadores Clásicos
if __name__ == "__main__":
    # Configuraciones específicas
    tree_config = {
        'max_depth': None,
        'min_samples_leaf': 5
    }
    svm_config = {
        'C': 1.0,
        'gamma': 'scale'
    }
    # Entrenar y evaluar Árbol de Decisión
    tree_model, tree_scaler, tree_pca, th_tree = entrenar_modelo_clasico(
        tree_config, X_train_full, y_train_full, variance_threshold=0.95, tipo='tree'
    )
    print("Umbral Árbol de Decisión:", th_tree)
    evaluar_modelo_clasico(tree_model, tree_scaler, tree_pca, X_test, y_test, th_tree, nombre='Árbol de Decisión')

    # Entrenar y evaluar SVM (RBF)
    svm_model, svm_scaler, svm_pca, th_svm = entrenar_modelo_clasico(
        svm_config, X_train_full, y_train_full, variance_threshold=0.95, tipo='svm'
    )
    print("Umbral SVM RBF:", th_svm)
    evaluar_modelo_clasico(svm_model, svm_scaler, svm_pca, X_test, y_test, th_svm, nombre='SVM RBF')

    # (Opcional) Guardar modelos y resultados similar a la Parte 2 del NN
    resultados = {
        'tree': {'auc_pr': average_precision_score(y_test, tree_model.predict_proba(tree_pca.transform(tree_scaler.transform(X_test)))[:,1])},
        'svm': {'auc_pr': average_precision_score(y_test, svm_model.predict_proba(svm_pca.transform(svm_scaler.transform(X_test)))[:,1])}
    }
    with open('resultados_clasificadores_classicos.json', 'w') as f:
        json.dump(resultados, f, indent=4)
    print("Resultados clásicos guardados en 'resultados_clasificadores_classicos.json'.")

