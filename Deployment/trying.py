
"""Predictive Maintenance con Optimización de Hiperparámetros para AUC-PR"""
import numpy as np
import pandas as pd
#import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay,
    precision_recall_curve,
    f1_score
)
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
from ConfigSpace.conditions import GreaterThanCondition, EqualsCondition
from smac.facade import HyperbandFacade
from smac import Scenario
import tensorflow as tf
from imblearn.combine import SMOTETomek


# --------------------------------------------
# 1. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
# Configurar paralelismo de TensorFlow
tf.config.threading.set_inter_op_parallelism_threads(10)
tf.config.threading.set_intra_op_parallelism_threads(10)

# --------------------------------------------
# 2. Carga y Preparación de Datos
# --------------------------------------------
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['burned_transformers'])
y = df_final['burned_transformers']

y.value_counts()

# Split inicial en train (80%) y test (20%) SIN BALANCEAR
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 3. Función de Preprocesamiento (Ahora Segura)
# --------------------------------------------

def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None, scaler=None):
    """
    Aplica SMOTE + Tomek y escalado SIN data leakage.
    
    - **Key Fix**: El scaler se ajusta SOLO a los datos originales (X_train_raw), 
      no a los datos balanceados por SMOTE.
    - X_val_raw: Datos de validación (no se balancean ni se usan en el fit del scaler)
    - scaler: Si se proporciona, se usa para transformar. Si no, se crea uno nuevo.
    """
    # 1. Ajustar scaler SOLO a datos originales (no balanceados)
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)  # Fit solo en datos originales
    
    # 2. Aplicar SMOTE + Tomek a los datos de entrenamiento
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    # 3. Transformar datos balanceados con scaler ya ajustado
    X_train_scaled = scaler.transform(X_train_bal)  # Usa scaler entrenado en X_train_raw
    
    # 4. Escalar validación (si existe) con el mismo scaler
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    return X_train_scaled, y_train_bal, X_val_scaled, scaler

# --------------------------------------------
# 4. Arquitectura de la Red Neuronal
# --------------------------------------------


def focal_loss(alpha=0.25, gamma=2.0):
    """Implementación de Focal Loss para clasificación binaria."""
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        loss = alpha * K.pow((1 - p_t), gamma) * bce
        return K.mean(loss)
    return loss


def create_model(best_config, input_dim):
    """
    Crea y devuelve un modelo utilizando los mejores hiperparámetros.
    """
    model = tf.keras.Sequential()
    
    num_layers = best_config['num_layers']
    
    for i in range(num_layers):
        units = best_config[f"units_{i+1}"]
        activation = best_config[f"activation_{i+1}"]
        
        # Regularización L1 y L2
        l1 = best_config['l1'] if best_config['use_l1'] else 0.0
        l2 = best_config['l2'] if best_config['use_l2'] else 0.0
        regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
        
        # Capa densa con regularización
        if i == 0:
            model.add(tf.keras.layers.Dense(units, activation=activation, input_dim=input_dim, kernel_regularizer=regularizer))
        else:
            model.add(tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=regularizer))
        
        # Dropout
        dropout = best_config['dropout']
        model.add(tf.keras.layers.Dropout(dropout))
    
    # Capa de salida
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Elección del optimizador
    optimizer_choice = best_config['optimizer']
    if optimizer_choice == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_config['learning_rate'])
    elif optimizer_choice == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_config['learning_rate'])
    elif optimizer_choice == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=best_config['learning_rate'])

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model




# --------------------------------------------
# 5. Función Objetivo para SMAC (Corregida)
# --------------------------------------------

def find_best_threshold(y_true, y_pred):
    """Encuentra el umbral óptimo que maximiza el F1-score basado en la curva Precision-Recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Evita divisiones por 0
    best_threshold = thresholds[np.nanargmax(f1_scores)]
    return best_threshold

def objective_function(config, seed, budget):
    """
    Función objetivo modificada:
    - Balanceo y escalado DENTRO de cada fold.
    - Validación con datos originales (no balanceados).
    - Métricas: AUC-PR y F1-score.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    f1_scores = []
    
    # Para guardar las métricas de cada fold
    auc_pr_train_all_folds = []
    auc_pr_val_all_folds = []
    f1_train_all_folds = []
    f1_val_all_folds = []
    thresholds_all_folds = []  # Lista para guardar los umbrales óptimos

    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        # Datos originales sin balancear
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # Preprocesar SOLO el conjunto de entrenamiento del fold
        X_train_scaled, y_train_bal, X_val_scaled, _ = preprocesar_datos(
            X_train_raw, y_train_raw, X_val_raw
        )

        # Entrenar modelo
        model = create_model(config)
        class_weight = {0: 1.0, 1: 30310/1436}
        
        history = model.fit(
            X_train_scaled, y_train_bal,
            validation_data=(X_val_scaled, y_val_raw),
            epochs=int(budget),
            batch_size=config["batch_size"],
            verbose=0,
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True, mode='min'
            )]
        )
       
        # Extraer métricas de AUC-PR y F1 desde el history
        auc_pr_train_all_folds.append(history.history['auc'])
        auc_pr_val_all_folds.append(history.history['val_auc'])

        f1_train_all_folds.append(history.history['f1'])
        f1_val_all_folds.append(history.history['val_f1'])

        # Predicciones
        y_val_pred = model.predict(X_val_scaled, verbose=0).ravel()

        # AUC-PR
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)

        # Calcular el umbral óptimo y F1-score
        best_threshold = find_best_threshold(y_val_raw, y_val_pred)
        thresholds_all_folds.append(best_threshold)  # Guardar el umbral óptimo
        y_val_binary = (y_val_pred >= best_threshold).astype(int)
        f1 = f1_score(y_val_raw, y_val_binary)
        f1_scores.append(f1)

    # Promedio de las métricas
    return (
        (1 - np.mean(auc_pr_scores)) + (1 - np.mean(f1_scores)) / 2,  # ← Métrica de optimización
        thresholds_all_folds,  # ← Umbrales óptimos
        auc_pr_train_all_folds,
        auc_pr_val_all_folds,
        f1_train_all_folds,
        f1_val_all_folds)


def obtener_umbral_optimo(thresholds_all_folds, y_val_raw, y_val_pred):
    """
    Devuelve el umbral óptimo promedio o el mejor umbral basado en el rendimiento en los folds.
    """
    # Usar el umbral promedio de los diferentes folds
    best_threshold_avg = np.mean(thresholds_all_folds)

    # O, alternativamente, buscar el umbral que maximiza el F1-score globalmente
    #best_threshold = find_best_threshold(y_val_raw, y_val_pred)
    
    return best_threshold_avg  # O retornar `best_threshold` si prefieres uno específico



# --------------------------------------------
# 6. Espacio de Búsqueda de Hiperparámetros
# --------------------------------------------

def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 12

    # Hiperparámetros principales
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.05, 0.5)  # Rango dropout
    batch_size = CategoricalHyperparameter('batch_size', [32, 64, 128, 256])
    optimizer = CategoricalHyperparameter('optimizer', ['adam', 'rmsprop', 'sgd'])  # Nuevo
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size, optimizer])

    # Regularización L1 y L2
    l1_reg = UniformFloatHyperparameter('l1', 1e-6, 1e-2, log=True)  # Regularización L1
    l2_reg = UniformFloatHyperparameter('l2', 1e-6, 1e-2, log=True)  # Regularización L2
    use_l1 = CategoricalHyperparameter("use_l1", [True, False])  # Usar o no L1
    use_l2 = CategoricalHyperparameter("use_l2", [True, False])  # Usar o no L2

    cs.add_hyperparameters([l1_reg, l2_reg, use_l1, use_l2])

    # Condiciones para usar L1 o L2 (opcional, pero recomendado)
    cs.add_condition(EqualsCondition(l1_reg, use_l1, True))  # Si use_l1 es True, se usa l1_reg
    cs.add_condition(EqualsCondition(l2_reg, use_l2, True))  # Si use_l2 es True, se usa l2_reg

    # Hiperparámetros condicionales por capa
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])

        # Condiciones para capas adicionales (asegura que solo existan si num_layers >= i)
        cs.add_condition(EqualsCondition(units, num_layers, i))  # Solo se asigna unidades si la capa existe
        cs.add_condition(EqualsCondition(activation, num_layers, i))  # Solo se asigna activación si la capa existe

        # Asegurar que las capas extra no tengan valores si num_layers < i
        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))

    return cs


# --------------------------------------------
# 7. Optimización con SMAC
# --------------------------------------------
def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=300,  
        min_budget=5,
        max_budget=50,
        n_workers=10
    )
    
    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )
    
    return smac.optimize()

# --------------------------------------------
# 8. Entrenamiento Final 
# --------------------------------------------

def entrenar_modelo_final(best_config, X_train_full, y_train_full):
    """
    Entrena el modelo final con los mejores hiperparámetros.
    - Primero divide en train/validation.
    - Luego aplica SMOTE solo al conjunto de train.
    """
    # 1. Dividir en train y validation (sin aplicar SMOTE todavía)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )

    # 2. Aplicar SMOTE solo al conjunto de train
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)

    # 3. Escalar los datos (ajustar el scaler solo en el train balanceado)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)

    # 4. Crear y entrenar el modelo
    model = create_model(best_config, input_dim=X_train_scaled.shape[1])  # Ahora incluye el optimizador
    class_weight = {0: 1.0, 1: np.sqrt(30310 / 1436)}  # Ajustar según el desbalanceo

    history = model.fit(
        X_train_scaled, y_train_bal,
        validation_data=(X_val_scaled, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )]
    )

    return model, history, scaler


# --------------------------------------------
# 9. Evaluación
# --------------------------------------------
def evaluar_modelo(model, X_test_final, y_test, thresholds_all_folds):
    """Evalúa el modelo en los datos de prueba y genera métricas clave."""
    y_pred_prob = model.predict(X_test_final)

    # Obtener el umbral óptimo (promedio de los folds)
    threshold = obtener_umbral_optimo(thresholds_all_folds, y_test, y_pred_prob)

    # AUC-PR y AUC-ROC
    auc_pr = average_precision_score(y_test, y_pred_prob)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n🔹 AUC-PR en Test: {auc_pr:.4f}")
    print(f"🔹 AUC-ROC en Test: {auc_roc:.4f}")

    # Matriz de confusión y reporte de clasificación
    y_pred = (y_pred_prob >= threshold).astype(int)

    print("\n Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # Visualizar curva Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.title(f'Curva Precision-Recall (AUC-PR = {auc_pr:.2f})')
    plt.show()


def plot_metrics(auc_pr_train, auc_pr_val, f1_train, f1_val):
    """Grafica la evolución de AUC-PR y F1-score en los folds."""
    plt.figure(figsize=(12, 5))

    # Gráfico AUC-PR
    plt.subplot(1, 2, 1)
    plt.plot(np.mean(auc_pr_train, axis=0), label="AUC-PR Train", linestyle="--", color="blue")
    plt.plot(np.mean(auc_pr_val, axis=0), label="AUC-PR Val", color="blue")
    plt.xlabel("Épocas")
    plt.ylabel("AUC-PR")
    plt.legend()
    plt.title("Curva AUC-PR")

    # Gráfico F1-score
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(f1_train, axis=0), label="F1 Train", linestyle="--", color="green")
    plt.plot(np.mean(f1_val, axis=0), label="F1 Val", color="green")
    plt.xlabel("Épocas")
    plt.ylabel("F1-score")
    plt.legend()
    plt.title("Evolución del F1-score")

    plt.show()

# --------------------------------------------
# 9. Despliegue
# --------------------------------------------

# Optimización de hiperparámetros

best_config = run_bohb_with_smac()
print("\nMejor configuración encontrada:")
for key, value in best_config.items():
    print(f"{key}: {value}")


# Modelo
# Conjunto de entrenamiento completo
X_train_final, y_train_final, scaler_final = preprocesar_datos(X_train_full, y_train_full, scaler_fit=True)
# conjunto de prueba con el scaler final
X_test_final = scaler_final.transform(X_test)

# modelo final con la mejor configuración encontrada
final_model = create_model(best_config)
# Callback de early stopping para el entrenamiento final
early_stop_final = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=5, restore_best_weights=True, mode='max', verbose=1)

# Entrenar el modelo final 
final_model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=best_config["batch_size"],
    verbose=1,
    callbacks=[early_stop_final]
)


# --------------------------------------------
# 10. Guardado de modelo
# --------------------------------------------

final_model.save('Modelo.h5')

import json
import joblib


best_config_dict = best_config.get_dictionary()

with open('hiperparametros.json', 'w') as f:
    json.dump(best_config_dict, f, indent=4)
    
joblib.dump(scaler_final, 'scaler.pkl')
