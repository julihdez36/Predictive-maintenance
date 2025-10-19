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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


os.listdir()

df = pd.read_csv('Data\df_entrenamiento_final.csv')

df.columns


# Función para separar entrenamiento/validación

def train_test_split_svm(df, target, n_train_total):
    """
    Divide el dataset en entrenamiento y validación siguiendo la lógica del artículo:
    - Todos los positivos (quemados) en entrenamiento
    - Una muestra de negativos (buenos) para completar n_train_total
    """
    burned = df[df[target] == 1]
    good   = df[df[target] == 0]

    n_train_burned = len(burned)
    n_train_good   = n_train_total - n_train_burned
    
    train_good_sample = good.sample(n=n_train_good, random_state=123)
    
    train_df = pd.concat([burned, train_good_sample])
    validation_df = df.drop(train_df.index)
    
    return train_df, validation_df

def scale_data(train_df, validation_df, target):
    num_vars = train_df.select_dtypes(include=np.number).columns.drop(target)
    
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    validation_scaled = validation_df.copy()
    
    train_scaled[num_vars] = scaler.fit_transform(train_df[num_vars])
    validation_scaled[num_vars] = scaler.transform(validation_df[num_vars])
    
    return train_scaled, validation_scaled

def train_svm(train_scaled, validation_scaled, target, kernel='rbf', cost=1.0, gamma='scale'):
    X_train = train_scaled.drop(columns=[target])
    y_train = train_scaled[target]
    
    X_val = validation_scaled.drop(columns=[target])
    y_val = validation_scaled[target]
    
    model = SVC(kernel=kernel, C=cost, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    print("Matriz de confusión:")
    print(confusion_matrix(y_val, y_pred))
    print("\nReporte de clasificación:")
    print(classification_report(y_val, y_pred, digits=4))
    
    return model

def grid_search_svm(train_scaled, target):
    X_train = train_scaled.drop(columns=[target])
    y_train = train_scaled[target]
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 'scale'],
        'kernel': ['rbf']
    }
    
    grid = GridSearchCV(SVC(probability=True), param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)
    
    print("Mejores parámetros:", grid.best_params_)
    return grid.best_estimator_


df.columns
# División entrenamiento/validación
train19, val19 = train_test_split_svm(df[df.year == 2019], target='burned_transformers', n_train_total=2417)

# Escalado
train19_scaled, val19_scaled = scale_data(train19, val19, target='burned_transformers')

# Entrenamiento SVM
svm19 = train_svm(train19_scaled, val19_scaled, target='burned_transformers')

# Opcional: buscar mejores hiperparámetros
# best_svm19 = grid_search_svm(train19_scaled, target='burned_transformers')

