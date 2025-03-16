import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
)
from sklearn.tree import DecisionTreeClassifier

# Importar el clasificador RUSBoost desde imblearn.ensemble
from imblearn.ensemble import RUSBoostClassifier

# 1. Cargar datos
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']

# Dividir en conjuntos de entrenamiento y prueba (estratificando para mantener la distribución)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Preprocesamiento: Escalado y PCA
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA para conservar el 95% de la varianza
pca = PCA(n_components=0.95, random_state=42)
X_train_reducido = pca.fit_transform(X_train_scaled)
X_test_reducido = pca.transform(X_test_scaled)

# 3. Definir y entrenar el clasificador RUSBoost
# Se utiliza un árbol de decisión "stump" (max_depth=1) como clasificador débil
rusboost = RUSBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
rusboost.fit(X_train_reducido, y_train)

# 4. Evaluación del modelo
# Obtener probabilidades y predicciones
y_pred_prob = rusboost.predict_proba(X_test_reducido)[:, 1]
y_pred = rusboost.predict(X_test_reducido)

# Calcular métricas
auc_pr = average_precision_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUC-PR: {auc_pr:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Curva de Precisión-Recall
disp = PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
plt.title(f"Curva de Precisión-Recall (AUC-PR = {auc_pr:.4f})")
plt.savefig("precision_recall_curve_rusboost.png", dpi=300, bbox_inches='tight')
plt.show()

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, cbar=False,
            xticklabels=['No Fallo', 'Fallo'], yticklabels=['No Fallo', 'Fallo'])
plt.title("Matriz de Confusión - RUSBoost")
plt.ylabel("Etiqueta Verdadera")
plt.xlabel("Etiqueta Predicha")
plt.savefig("confusion_matrix_rusboost.png", dpi=300, bbox_inches='tight')
plt.show()
