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
    f1_score
)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l1, l2
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
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Función de pérdida focal para abordar el desbalanceo de clases.
    """
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        return K.mean(alpha * K.pow(1 - p_t, gamma) * bce)
    return loss

def create_model(best_config, input_dim):
    """
    Crea y compila un modelo Keras de acuerdo con la configuración de hiperparámetros.
    """
    model = Sequential()
    num_layers = best_config['num_layers']
    
    # Se agregan las capas ocultas
    for i in range(num_layers):
        units = best_config[f"units_{i+1}"]
        activation = best_config[f"activation_{i+1}"]
        # Se aplican regularizadores si se han activado
        l1_val = best_config['l1'] if best_config['use_l1'] else 0.0
        l2_val = best_config['l2'] if best_config['use_l2'] else 0.0
        regularizer = tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)
        
        # La primera capa requiere especificar input_shape
        if i == 0:
            model.add(Dense(units, kernel_regularizer=regularizer, input_shape=(input_dim,)))
        else:
            model.add(Dense(units, kernel_regularizer=regularizer))
            
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(best_config['dropout']))
    
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    # Selección del optimizador
    lr = best_config['learning_rate']
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr)
    }
    optimizer = optimizers.get(best_config['optimizer'], Adam(learning_rate=lr))
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

# --------------------------------------------
# 5. Funciones Auxiliares para Evaluación
# --------------------------------------------
def find_best_threshold(y_true, y_pred):
    """
    Encuentra el umbral que maximiza el F1 score a partir de la curva de precisión-recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

def objective_function(config, seed, budget):
    """
    Función objetivo para SMAC. Se realiza validación cruzada estratificada
    y se retorna 1 - AUC-PR promedio, ya que SMAC minimiza la función objetivo.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        X_train_scaled, y_train_bal, X_val_scaled, _ = preprocesar_datos(X_train_raw, y_train_raw, X_val_raw)
        model = create_model(config, X_train_scaled.shape[1])
        
        model.fit(
            X_train_scaled, y_train_bal,
            validation_data=(X_val_scaled, y_val_raw),
            epochs=int(budget),
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        y_val_pred = model.predict(X_val_scaled, verbose=0).ravel()
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)
    
    return 1 - np.mean(auc_pr_scores)  # SMAC minimiza este valor

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
    """
    Entrena el modelo final utilizando la mejor configuración encontrada y retorna
    el modelo, el escalador y el umbral óptimo basado en la curva Precision-Recall.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    # Balanceo del conjunto de entrenamiento
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    # Escalado de características
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    # Creación y entrenamiento del modelo
    model = create_model(best_config, X_train_scaled.shape[1])
    model.fit(
        X_train_scaled, y_train_bal,
        validation_data=(X_val_scaled, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'
        )]
    )
    
    # Cálculo del umbral óptimo basado en la curva Precision-Recall
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
    
    print(f"AUC-PR: {average_precision_score(y_test, y_pred_prob):.4f}")
    print(classification_report(y_test, y_pred))
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.show()


import json
import joblib

# --------------------------------------------
# 9. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    # Se ejecuta la optimización de hiperparámetros
    best_config = run_bohb_with_smac()

    # guardemos configuracion
    best_config_dict = best_config.get_dictionary()

    with open('hiperparametros.json', 'w') as f:
        json.dump(best_config_dict, f, indent=4)
    
    # Entrenamiento final con la mejor configuración
    final_model, scaler, threshold = entrenar_modelo_final(best_config, X_train_full, y_train_full)

    # guardar el modelo final
    final_model.save('modelo_fitted.h5') 
    joblib.dump(scaler, 'scaler.pkl')
    
    # Evaluación en el conjunto de test
    #X_test_scaled = scaler.transform(X_test)
    #evaluar_modelo(final_model, X_test_scaled, y_test, threshold)
    


        
    
