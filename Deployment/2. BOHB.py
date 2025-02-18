# ============================================
# Parte 1: Búsqueda de la Mejor Configuración
# ============================================
"""
Este script realiza la optimización de hiperparámetros (incluyendo la integración de PCA para conservar el 95% de la varianza)
usando BOHB (a través de SMAC) y ConfigSpace. Se utiliza validación con StratifiedKFold.
"""

# 1. Importaciones
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, precision_recall_curve

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from smac.facade import HyperbandFacade
from smac import Scenario
from imblearn.combine import SMOTETomek

# 2. Configuración Inicial y Datos
tf.random.set_seed(42)
np.random.seed(42)

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']

# Dividir en entrenamiento y prueba (para la optimización usaremos solo el entrenamiento)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Funciones para Preprocesamiento y PCA
def preprocesar_datos_con_pca(X_train_scaled, y_train, pca, aplicar_smote=False):
    """
    Transforma los datos escalados usando PCA y, opcionalmente,
    aplica SMOTETomek en el espacio reducido.
    """
    X_train_reducido = pca.transform(X_train_scaled)
    if aplicar_smote:
        smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
        X_train_reducido, y_train_bal = smote_tomek.fit_resample(X_train_reducido, y_train)
    else:
        y_train_bal = y_train
    return X_train_reducido, y_train_bal

def entrenar_pca(X_train_scaled, variance_threshold=0.95):
    """
    Ajusta PCA sobre los datos escalados conservando el porcentaje de varianza especificado.
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(X_train_scaled)
    return pca

# 4. Callback OneCycleLR
class OneCycleLR(tf.keras.callbacks.Callback):
    def __init__(self, max_lr, epochs, start_lr=None, end_lr=None):
        super(OneCycleLR, self).__init__()
        self.epochs = epochs
        self.max_lr = max_lr
        self.start_lr = start_lr if start_lr is not None else max_lr / 10.0
        self.end_lr = end_lr if end_lr is not None else self.start_lr / 100.0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs * 0.3:
            lr = self.start_lr + (self.max_lr - self.start_lr) * (epoch / (self.epochs * 0.3))
        else:
            lr = self.max_lr - (self.max_lr - self.end_lr) * ((epoch - self.epochs * 0.3) / (self.epochs * 0.7))
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"\nEpoch {epoch+1}: OneCycleLR setting learning rate to {lr:.6f}")

# 5. Función de Pérdida Focal
def focal_loss(alpha=0.95, gamma= 5):
    def loss(y_true, y_pred):
        y_true = K.cast(y_true, K.floatx())
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        bce = K.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_factor = K.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_factor * focal_factor * bce)
    return loss

# 6. Definición de la Arquitectura del Modelo
def create_model(best_config, input_dim):
    """
    Crea el clasificador final basado en la configuración y la dimensión de entrada.
    """
    model = Sequential()
    num_layers = best_config['num_layers']
    lr = float(best_config['learning_rate'])
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
    model.add(Dense(1, activation='sigmoid'))
    
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr, momentum=0.9, nesterov=True),
        'nadam': Nadam(learning_rate=lr)
    }
    optimizer = optimizers.get(best_config['optimizer'], Adam(learning_rate=lr))
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.95, gamma=5),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

def find_best_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

# 7. Definición del Espacio de Hiperparámetros con ConfigSpace
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
    
    # Hiperparámetros por capa
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])
        if i > 1:
            cs.add_condition(InCondition(child=units, parent=num_layers, values=list(range(i, max_layers+1))))
            cs.add_condition(InCondition(child=activation, parent=num_layers, values=list(range(i, max_layers+1))))
    
    return cs

# 8. Función Objetivo para la Optimización (usando PCA)

def objective_function(config, seed, budget):
    """
    Realiza validación con StratifiedKFold integrando PCA (95% de varianza) en el pipeline.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    fold_histories = []
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # Ajustar PCA (95% de varianza)
        pca = entrenar_pca(X_train_scaled, variance_threshold=0.95)
        
        # Aplicar SMOTETomek en el espacio reducido (solo entrenamiento)
        X_train_reducido, y_train_bal = preprocesar_datos_con_pca(X_train_scaled, y_train, pca, aplicar_smote=True)
        X_val_reducido = pca.transform(X_val_scaled)
        
        input_dim = X_train_reducido.shape[1]
        model = create_model(config, input_dim=input_dim)
        
        # Calcular pesos de clases
        class_weights = {0: 1, 1: 10}
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=5, restore_best_weights=True),
            OneCycleLR(max_lr=config['learning_rate'] * 10, epochs=int(budget))
        ]
        
        history = model.fit(
            X_train_reducido, y_train_bal,
            validation_data=(X_val_reducido, y_val),
            epochs=int(budget),
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        fold_histories.append({k: list(map(float, v)) for k, v in history.history.items()})
        y_val_pred = model.predict(X_val_reducido, verbose=0).ravel()
        auc_pr = average_precision_score(y_val, y_val_pred)
        auc_pr_scores.append(auc_pr)
    
    return 1 - np.mean(auc_pr_scores)

# 9. Optimización con BOHB
def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=300,
        min_budget=5,
        max_budget=70,
        n_workers=10
    )
    
    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )
    
    best_config = smac.optimize()
    # Convertir a diccionario
    best_config_dict = best_config.get_dictionary()
    print("Mejor configuración encontrada:")
    print(best_config_dict)
    # Puedes guardar el diccionario si lo deseas:
    with open("hiperparametros.json", "w") as f:
        json.dump(best_config_dict, f, indent=4)
    return best_config_dict


if __name__ == "__main__":
    best_config = run_bohb_with_smac()

    best_config_dict = best_config.get_dictionary()
    
    with open("hiperparametros.json", "w") as f:
        json.dump(best_config_dict, f, indent=4)
    print("La configuración de hiperparámetros se ha guardado en 'hiperparametros.json'.")
