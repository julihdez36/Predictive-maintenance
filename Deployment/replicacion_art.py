# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 22:17:58 2025

@author: Julian
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


os.listdir()

df = pd.read_csv('Data\df_entrenamiento_final.csv')

df.columns


"""
Separar entrenamiento/validación

Divide el dataset en entrenamiento y validación siguiendo la lógica del
 artículo:
     - Todos los positivos (quemados) en entrenamiento
     - Una muestra de negativos (buenos) para completar n_train_total
"""

entrenamiento19 = df[df['year'] == 2019]
burned19 = entrenamiento19[entrenamiento19['burned_transformers']== 1]
train_19_good = entrenamiento19.sample(n = 1610, random_state = 42)

train19 = pd.concat([burned19, train_19_good])

# Hagamos lo mismo para el 2020 ############################

entrenamiento20 = df[df['year'] == 2020]
burned20 = entrenamiento20[entrenamiento20['burned_transformers']== 1]
train_20_good = entrenamiento20.sample(n = 1610, random_state = 42)

train20 = pd.concat([burned20, train_20_good])

'''
En el articulo no parecen decir que lo escalaron, hagamoslo sin escalar
y posteriormente escalemos a ver si algo cambia

Para ello, ajustemos primero el set para el entrenamiento
'''
y_train19 = train19['burned_transformers']
X_train19 = train19.drop(columns = ['burned_transformers', 'year','eens_kwh' ])

y_test20 = train20['burned_transformers']
X_train20 = train20.drop(columns = ['burned_transformers', 'year','eens_kwh'])

# Entrenamos con 2019 submuestreada y testeamos con 2020 submuestreada

linear_model = LinearSVC(C=3, random_state=42) 
model = CalibratedClassifierCV(linear_model, method='sigmoid', cv=5)
model.fit(X_train19, y_train19)

y_pred = model.predict(X_train20)

print(confusion_matrix(y_test20, y_pred))
print(classification_report(y_test20, y_pred, digits=4))


# Testeamos sin submuestreo en el 2020 ############################

x_test = entrenamiento20.drop(columns = ['burned_transformers', 'year','eens_kwh'])
y_pred = model.predict(x_test)
y_test = entrenamiento20['burned_transformers']

y_pred = model.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=4))

    

'''
Escalemos los datos para ver si mejoran los resultados
'''

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train19)
x_test_scaled = scaler.transform(x_test)

linear_model = LinearSVC(C=3, random_state=42) 
model = CalibratedClassifierCV(linear_model, method='sigmoid', cv=5)
model.fit(x_train_scaled, y_train19)

y_pred = model.predict(x_test)

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, digits=4))

'''
Vemos que con el escalado mejora, alcanzando un recall de dl 44.36%,
es decir, mejora la tasa de descurbrimiento de los fallos.


'''


X_train19.columns

# Escalar datos
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train19)
x_test_scaled = scaler.transform(x_test)

# 1️⃣ Entrenar LinearSVC directo (solo para coeficientes)
linear_model = LinearSVC(C=3, random_state=42)
linear_model.fit(x_train_scaled, y_train19)

# Extraer coeficientes
importances = linear_model.coef_[0]

# Asociar con nombres de variables y ordenar
feature_importance = pd.Series(importances, index=X_train19.columns)
feature_importance = feature_importance.abs().sort_values(ascending=False)

# 2️⃣ Calibrar el modelo (para predicción final si quieres)
calibrated_model = CalibratedClassifierCV(linear_model, method='sigmoid', cv=5)
calibrated_model.fit(x_train_scaled, y_train19)

# Predicciones finales
y_pred = calibrated_model.predict(x_test_scaled)

# 3️⃣ Graficar TOP 15 características
top_n = 18
feature_importance.head(top_n).plot(kind='bar', figsize=(10, 5))
plt.title('Importancia de variables según LinearSVC')
plt.ylabel('Magnitud del coeficiente (|peso|)')
plt.xlabel('Variables')
plt.tight_layout()
plt.show()


# Rescatando el signo ########################

# 1️⃣ Escalar datos
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train19)
x_test_scaled = scaler.transform(x_test)

# 2️⃣ Entrenar LinearSVC para extraer coeficientes
linear_model = LinearSVC(C=3, random_state=42)
linear_model.fit(x_train_scaled, y_train19)

# 3️⃣ Extraer coeficientes con signo
coef_with_sign = pd.Series(linear_model.coef_[0], index=X_train19.columns)

# Crear columna con magnitud absoluta
feature_importance = coef_with_sign.abs().sort_values(ascending=False)

# 4️⃣ Seleccionar top N características
top_n = 18
top_features = feature_importance.head(top_n).index
top_coef_with_sign = coef_with_sign[top_features]

# 5️⃣ Asignar color: rojo si impulsa hacia fallo (coef > 0), azul si impulsa hacia no fallo (coef < 0)
colors = ['red' if w > 0 else 'blue' for w in top_coef_with_sign]

# 6️⃣ Graficar
plt.figure(figsize=(10, 5))
top_coef_with_sign.sort_values(ascending=False).plot(kind='bar', color=colors)

plt.title('Importancia y dirección de las variables según LinearSVC')
plt.ylabel('Coeficiente (con signo)')
plt.xlabel('Variables')

# Leyenda manual
plt.axhline(0, color='black', linewidth=0.8)
plt.text(0.5, max(top_coef_with_sign)*0.9, 'Rojo = aumenta prob. de falla', color='red')
plt.text(0.5, min(top_coef_with_sign)*0.9, 'Azul = reduce prob. de falla', color='blue')

plt.tight_layout()
plt.show()

# 7️⃣ Calibrar modelo (para uso final)
calibrated_model = CalibratedClassifierCV(linear_model, method='sigmoid', cv=5)
calibrated_model.fit(x_train_scaled, y_train19)

# Predicción final
y_pred = calibrated_model.predict(x_test_scaled)
