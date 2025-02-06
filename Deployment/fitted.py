"""Predictive Maintenance con Optimización de Hiperparámetros para AUC-PR"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    PrecisionRecallDisplay,
    precision_recall_curve,
    f1_score,
    recall_score,
    precision_score
)

import tensorflow as tf
import tensorflow.keras.backend as K
# from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from ConfigSpace import (
    ConfigurationSpace,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter
)
from ConfigSpace.conditions import GreaterThanCondition, EqualsCondition
from smac.facade import HyperbandFacade
from smac import Scenario

from imblearn.combine import SMOTETomek
import json
import joblib

# --------------------------------------------
# 1. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
tf.config.threading.set_inter_op_parallelism_threads(10)
tf.config.threading.set_intra_op_parallelism_threads(10)

# --------------------------------------------
# 2. Carga y Preparación de Datos
# --------------------------------------------
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['burned_transformers'])
y = df_final['burned_transformers']

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------
# 3. Función de Preprocesamiento
# --------------------------------------------
def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None, scaler=None):
    """
    Escala los datos utilizando RobustScaler y aplica SMOTETomek para balancear
    el conjunto de entrenamiento.
    """
    if scaler is None:
        scaler = RobustScaler()
        scaler.fit(X_train_raw)
    
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    return X_train_scaled, y_train_bal, X_val_scaled, scaler

# --------------------------------------------
# 4. Arquitectura de la Red Neuronal y Funciones de Pérdida
# --------------------------------------------

def focal_loss(alpha=0.5, gamma=2.5):
    """
    Función de pérdida focal optimizada para datos desbalanceados.
    
    Parámetros:
        alpha (float): Factor de ajuste para la clase minoritaria (0 < alpha < 1).
        gamma (float): Factor de modulación que ajusta la penalización de los ejemplos bien clasificados.
    
    Retorna:
        Función de pérdida focal que se puede utilizar en `model.compile(loss=focal_loss())`.
    """
    def loss(y_true, y_pred):
        # Convertir a float
        y_true = K.cast(y_true, K.floatx())
        
        # Asegurar estabilidad numérica
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calcular la pérdida binaria estándar
        bce = K.binary_crossentropy(y_true, y_pred)
        
        # Probabilidad de la clase verdadera
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Factor de modulación (penaliza ejemplos bien clasificados)
        focal_factor = K.pow(1.0 - p_t, gamma)
        
        # Aplicar alpha solo a la clase minoritaria
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Pérdida focal final
        return K.mean(alpha_factor * focal_factor * bce)
    
    return loss

def create_model(best_config, input_dim):
    """
    Crea y compila un modelo Keras con la configuración dada.
    """
    model = Sequential()
    num_layers = best_config['num_layers']
    
    # Regularización
    l1_val = best_config.get('l1', 0.0) if best_config.get('use_l1', False) else 0.0
    l2_val = best_config.get('l2', 0.0) if best_config.get('use_l2', False) else 0.0
    regularizer = tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)
    
    for i in range(num_layers):
        units = best_config[f"units_{i+1}"]
        activation = best_config[f"activation_{i+1}"]
        
        if i == 0:
            model.add(Dense(units, kernel_regularizer=regularizer, input_shape=(input_dim,), use_bias=False))
        else:
            model.add(Dense(units, kernel_regularizer=regularizer, use_bias=False))
            
        model.add(BatchNormalization())
        model.add(Activation(activation))
        
        if best_config["dropout"] > 0:
            model.add(Dropout(best_config["dropout"]))
    
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    lr = best_config['learning_rate']
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    }
    optimizer = optimizers.get(best_config['optimizer'], Adam(learning_rate=lr))
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.5, gamma=2.5),  # Valores ajustados
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model


# -------------------------------
# 4.1. Callback: Scheduler de Learning Rate
# -------------------------------
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_auc_pr',  # O puedes usar 'val_loss' u otra métrica
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# --------------------------------------------
# 5. Funciones Auxiliares para Evaluación
# --------------------------------------------
def find_best_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

global_history = []   # Para almacenar historiales de entrenamiento
global_thresholds = []  # Para almacenar umbrales

def objective_function(config, seed, budget):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores, thresholds_all_folds, history_all_folds = [], [], []
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        X_train_scaled, y_train_bal, X_val_scaled, _ = preprocesar_datos(X_train_raw, y_train_raw, X_val_raw)
        
        model = create_model(config, input_dim=X_train_scaled.shape[1])
        
        # Cálculo de pesos para las clases (opcional)
        class_weight = {
            0: len(y_train_bal) / (2 * sum(y_train_bal == 0)),
            1: len(y_train_bal) / (2 * sum(y_train_bal == 1))
        }
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            reduce_lr  # Se agrega el scheduler de LR
        ]
        
        history = model.fit(
            X_train_scaled, y_train_bal,
            validation_data=(X_val_scaled, y_val_raw),
            epochs=int(budget),
            batch_size=config["batch_size"],
            verbose=0,
            class_weight=class_weight,
            callbacks=callbacks
        )
        
        y_val_pred = model.predict(X_val_scaled, verbose=0).ravel()
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)
        
        best_threshold = find_best_threshold(y_val_raw, y_val_pred)
        thresholds_all_folds.append(best_threshold)
        
        history_all_folds.append(history.history)
    
    global_history.append({"config": config, "history": history_all_folds})
    global_thresholds.append(np.mean(thresholds_all_folds))
    
    return 1 - np.mean(auc_pr_scores)

# --------------------------------------------
# 6. Espacio de Búsqueda de Hiperparámetros
# --------------------------------------------
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 12

    # Hiperparámetros principales
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.05, 0.5)
    batch_size = CategoricalHyperparameter('batch_size', [32, 64, 128, 256])
    optimizer = CategoricalHyperparameter('optimizer', ['adam', 'rmsprop', 'sgd'])
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size, optimizer])

    # Regularización
    l1_reg = UniformFloatHyperparameter('l1', 1e-6, 1e-2, log=True)
    l2_reg = UniformFloatHyperparameter('l2', 1e-6, 1e-2, log=True)
    use_l1 = CategoricalHyperparameter("use_l1", [True, False])
    use_l2 = CategoricalHyperparameter("use_l2", [True, False])
    cs.add_hyperparameters([l1_reg, l2_reg, use_l1, use_l2])
    
    cs.add_condition(EqualsCondition(l1_reg, use_l1, True))
    cs.add_condition(EqualsCondition(l2_reg, use_l2, True))
    
    # Hiperparámetros por capa: omitimos la condición para la primera capa
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])
        # Para la primera capa no se necesita condicionar, pues num_layers siempre es >= 1
        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))
    
    return cs

# --------------------------------------------
# 7. Optimización con SMAC (HyperbandFacade)
# --------------------------------------------
def run_bohb_with_smac():
    """
    Ejecuta la optimización de hiperparámetros utilizando SMAC con la estrategia Hyperband.
    """
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
    
    best_config = smac.optimize()
    print("Mejor configuración encontrada:", best_config)
    return best_config

# --------------------------------------------
# 8. Entrenamiento Final y Evaluación
# --------------------------------------------
def entrenar_modelo_final(best_config, X_train_full, y_train_full):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    model = create_model(best_config, X_train_scaled.shape[1])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
        reduce_lr  # Se añade el scheduler
    ]
    
    model.fit(
        X_train_scaled, y_train_bal,
        validation_data=(X_val_scaled, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks
    )
    
    y_val_pred = model.predict(X_val_scaled).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, best_threshold

def evaluar_modelo(model, X_test, y_test, threshold):
    """
    Evalúa el modelo final mostrando el AUC-PR, el reporte de clasificación y
    la curva Precision-Recall.
    """
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    auc_pr = average_precision_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.show()


# --------------------------------------------
# 9. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    # Se ejecuta la optimización de hiperparámetros
    best_config = run_bohb_with_smac()

    # Entrenamiento final con la mejor configuración
    final_model, scaler, threshold = entrenar_modelo_final(best_config, X_train_full, y_train_full)
    
    # Evaluación en el conjunto de test
    X_test_scaled = scaler.transform(X_test)
    evaluar_modelo(final_model, X_test_scaled, y_test, threshold)
    
    # --------------------------------------------
    # 10. Registro del modelo y la configuración
    # --------------------------------------------
    
    # Convertir best_config a diccionario, según el objeto
    if hasattr(best_config, 'get_dictionary'):
        best_config_dict = best_config.get_dictionary()
    else:
        best_config_dict = best_config  # Asumir que ya es un diccionario

    # Verificar que los datos sean serializables a JSON
    # Si hay algún objeto complejo, es posible que necesites convertirlo a un formato básico (p. ej., str)
    with open('hiperparametros.json', 'w') as f:
        json.dump(best_config_dict, f, indent=4)
        
    # Guardar el modelo final y el escalador
    final_model.save('modelo_fitted.h5')
    joblib.dump(scaler, 'scaler.pkl')