"""Ajustes para mejorar la predicción de la clase minoritaria.
   
   1. Remplazo del escalador RobustScaler por QuantileTransformer con salida
      en distribución normal.
   2. StratifiedKFold en la función objetivo para la validación.
   3. Ajuste EarlyStopping basado en 'val_auc_pr' para evitar el sobreajuste de la
      clase mayoritaria.
   4. Callback OneCycleLR que ajusta dinámicamente la tasa de aprendizaje
      durante el entrenamiento.
   5. Se aclara que la optimización se realiza con BOHB (mediante HyperbandFacade de SMAC).
"""

# --------------------------------------------
# 1. Importaciones
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer  # Se usa QuantileTransformer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, precision_recall_curve, f1_score, recall_score, precision_score
)
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from smac.facade import HyperbandFacade
from smac import Scenario
from imblearn.combine import SMOTETomek

# --------------------------------------------
# 2. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
tf.config.threading.set_inter_op_parallelism_threads(10)
tf.config.threading.set_intra_op_parallelism_threads(10)

# --------------------------------------------
# 3. Carga y Preparación de Datos
# --------------------------------------------
def cargar_datos(url):
    """
    Carga los datos y los divide en conjunto de entrenamiento y prueba.
    """
    df_final = pd.read_csv(url)
    X = df_final.drop(columns=['burned_transformers'])
    y = df_final['burned_transformers']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# URL de los datos
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_2.csv'
X_train_full, X_test, y_train_full, y_test = cargar_datos(url)

# --------------------------------------------
# 4. Preprocesamiento de Datos con QuantileTransformer
# --------------------------------------------
def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None, scaler=None):
    """
    Escala los datos usando QuantileTransformer para transformar la distribución
    a una distribución normal. Esto puede ayudar a que la información de la clase
    minoritaria se preserve de manera más adecuada. Además, se aplica SMOTETomek
    para balancear el conjunto de entrenamiento.
    """
    # Se inicializa y ajusta el escalador si no se proporciona
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
    
    # Aplicar SMOTETomek para balancear las clases
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    # Escalar datos de entrenamiento y validación
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    return X_train_scaled, y_train_bal, X_val_scaled, scaler

# --------------------------------------------
# 5. Callback OneCycleLR para Ajuste Dinámico de la Tasa de Aprendizaje
# --------------------------------------------

class OneCycleLR(tf.keras.callbacks.Callback):
    """
    Callback que implementa la política OneCycleLR.
    Durante el entrenamiento:
      - En la primera fase (aproximadamente 30% de las épocas), la tasa de aprendizaje
        aumenta linealmente desde un valor inicial hasta un máximo.
      - En la segunda fase, la tasa disminuye linealmente hasta un valor final.
    """
    def __init__(self, max_lr, epochs, start_lr=None, end_lr=None):
        super(OneCycleLR, self).__init__()
        self.epochs = epochs
        self.max_lr = max_lr
        self.start_lr = start_lr if start_lr is not None else max_lr / 10.0
        self.end_lr = end_lr if end_lr is not None else self.start_lr / 100.0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs * 0.3:
            # Fase de incremento lineal
            lr = self.start_lr + (self.max_lr - self.start_lr) * (epoch / (self.epochs * 0.3))
        else:
            # Fase de decremento lineal
            lr = self.max_lr - (self.max_lr - self.end_lr) * ((epoch - self.epochs * 0.3) / (self.epochs * 0.7))
        
        # Actualizar la tasa de aprendizaje
        # Se intenta usar el método assign si está disponible
        if hasattr(self.model.optimizer.learning_rate, 'assign'):
            self.model.optimizer.learning_rate.assign(lr)
        else:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        print(f"\nEpoch {epoch+1}: OneCycleLR setting learning rate to {lr:.6f}")


# --------------------------------------------
# 6. Función de Pérdida Focal para Datos Desbalanceados
# --------------------------------------------
def focal_loss(alpha=0.5, gamma=2.5):
    """
    Implementa la función de pérdida focal, que reduce el peso de ejemplos fáciles
    y se enfoca en los ejemplos difíciles, ayudando a mejorar la predicción de la
    clase minoritaria.
    """
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

# --------------------------------------------
# 7. Arquitectura de la Red Neuronal
# --------------------------------------------
def create_model(best_config, input_dim):
    """
    Crea y compila el modelo Keras según la configuración de hiperparámetros.
    """
    model = Sequential()
    num_layers = best_config['num_layers']
    
    # Convertir el learning_rate a float para evitar que sea un string.
    lr = float(best_config['learning_rate'])
    
    # Configuración de la regularización (L1 y/o L2)
    l1_val = best_config.get('l1', 0.0) if best_config.get('use_l1', False) else 0.0
    l2_val = best_config.get('l2', 0.0) if best_config.get('use_l2', False) else 0.0
    regularizer = tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)
    
    # Creación de las capas ocultas
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
    
    # Capa de salida para clasificación binaria
    model.add(Dense(1, activation='sigmoid'))
    
    # Selección del optimizador usando el learning_rate convertido
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    }
    optimizer = optimizers.get(best_config['optimizer'], Adam(learning_rate=lr))
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.5, gamma=2.5),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model


# --------------------------------------------
# 8. Función para Buscar el Umbral Óptimo Basado en F1-score
# --------------------------------------------
def find_best_threshold(y_true, y_pred):
    """
    Calcula el umbral que maximiza el F1-score utilizando la curva precisión-recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

# --------------------------------------------
# 9. Definición del Espacio de Hiperparámetros con ConfigSpace
# --------------------------------------------
def get_configspace():
    """
    Define el espacio de búsqueda de hiperparámetros.
    Se utiliza InCondition para activar los hiperparámetros de cada capa 
    únicamente cuando 'num_layers' sea mayor o igual que el índice de la capa.
    """
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
        # Para cada capa se definen unidades y activación
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])
        
        # Activar estos hiperparámetros solo si num_layers >= i.
        # Se usa InCondition para ello:
        if i > 1:
            cs.add_condition(InCondition(child=units, parent=num_layers, values=list(range(i, max_layers+1))))
            cs.add_condition(InCondition(child=activation, parent=num_layers, values=list(range(i, max_layers+1))))
    
    return cs

# --------------------------------------------
# 10. Función Objetivo para la Optimización de Hiperparámetros
# (Se usa StratifiedKFold para la validación, sin train_test_split adicional)
# --------------------------------------------
global_history = []  # Almacena los historiales de entrenamiento de cada configuración

def objective_function(config, seed, budget):
    """
    Función objetivo utilizada por SMAC/BOHB. Realiza validación con StratifiedKFold.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    fold_histories = []  # Almacena el historial de cada fold
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # Preprocesar datos con QuantileTransformer y balanceo con SMOTETomek
        X_train_scaled, y_train_bal, X_val_scaled, _ = preprocesar_datos(X_train_raw, y_train_raw, X_val_raw)
        
        model = create_model(config, input_dim=X_train_scaled.shape[1])
        
        # Cálculo de pesos de clases para compensar el desbalance
        class_weight = {
            0: len(y_train_bal) / (2 * sum(y_train_bal == 0)),
            1: len(y_train_bal) / (2 * sum(y_train_bal == 1))
        }
        
        # Callbacks: EarlyStopping basado en 'val_auc_pr' y OneCycleLR para ajustar la tasa de aprendizaje
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=5, restore_best_weights=True),
            OneCycleLR(max_lr=config['learning_rate'] * 10, epochs=int(budget))
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
        
        # Convertir history a un formato serializable
        fold_histories.append({k: list(map(float, v)) for k, v in history.history.items()})
        
        # Evaluar el AUC-PR en el fold de validación
        y_val_pred = model.predict(X_val_scaled, verbose=0).ravel()
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)
    
    # Almacenar historial y métrica promedio para esta configuración
    global_history.append({
        "config": config,
        "history": fold_histories,
        "mean_auc_pr": np.mean(auc_pr_scores)
    })
    
    return 1 - np.mean(auc_pr_scores)

# --------------------------------------------
# 11. Optimización de Hiperparámetros con BOHB (SMAC)
# --------------------------------------------
def run_bohb_with_smac():
    """
    Ejecuta la optimización de hiperparámetros utilizando BOHB.
    """
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=500,
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
    print("Mejor configuración encontrada:", best_config)
    return best_config

# --------------------------------------------
# 12. Entrenamiento Final del Modelo con la Mejor Configuración
# --------------------------------------------
def entrenar_modelo(best_config, X_train_full, y_train_full):
    """
    Entrena el modelo final usando la mejor configuración.
    Se separa un 10% de los datos para validación (usando train_test_split) para monitorear el EarlyStopping.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    # Aplicar SMOTETomek y preprocesar con QuantileTransformer
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    model = create_model(best_config, input_dim=X_train_scaled.shape[1])
    
    # Callbacks: EarlyStopping y OneCycleLR para un entrenamiento de 100 épocas
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
        OneCycleLR(max_lr=best_config['learning_rate'] * 10, epochs=100)
    ]
    
    model.fit(
        X_train_scaled, y_train_bal,
        validation_data=(X_val_scaled, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks
    )
    
    # Se busca el umbral óptimo en la validación
    y_val_pred = model.predict(X_val_scaled).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, best_threshold

# --------------------------------------------
# 13. Evaluación Final del Modelo en el Conjunto de Prueba
# --------------------------------------------

def evaluar_modelo(model, X_test, y_test, threshold):
    """
    Evalúa el modelo final en el conjunto de prueba e imprime las métricas.
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
    
    # Graficar la curva de precisión-recall
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.title(f"Curva de Precisión-Recall (AUC-PR = {auc_pr:.4f})")
    
    # Guardar la gráfica en un archivo
    plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

 
# --------------------------------------------
# 14. Función para Graficar el Historial de Entrenamiento
# --------------------------------------------
def plot_training_history(history):
    """
    Grafica la pérdida y el AUC-PR durante el entrenamiento para un fold.
    """
    plt.figure(figsize=(12, 5))
    
    # Gráfico de la pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico del AUC-PR
    plt.subplot(1, 2, 2)
    plt.plot(history['auc_pr'], label='Train AUC-PR')
    plt.plot(history['val_auc_pr'], label='Validation AUC-PR')
    plt.title('AUC-PR durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('AUC-PR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("pérdida y AUC-PR.png", dpi=300, bbox_inches='tight')
    plt.show()

       
# --------------------------------------------
# 15. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    # Fase 1: Optimización de Hiperparámetros usando BOHB (ya se usa BOHB en este código)
    print("Iniciando optimización de hiperparámetros...")
    best_config = run_bohb_with_smac()
    print("Optimización completada. Mejor configuración encontrada:")
    print(best_config)

    # Fase 2: Entrenamiento final del modelo con la mejor configuración
    print("Entrenando el modelo final con la mejor configuración...")
    final_model, scaler, threshold = entrenar_modelo(best_config, X_train_full, y_train_full)
    
    # Evaluación en el conjunto de prueba
    X_test_scaled = scaler.transform(X_test)
    evaluar_modelo(final_model, X_test_scaled, y_test, threshold)
    
    # Visualización del historial de entrenamiento para cada configuración evaluada
    for i, config_history in enumerate(global_history):
        print(f"\nConfiguración {i + 1}:")
        print(f"Hiperparámetros: {config_history['config']}")
        print(f"AUC-PR promedio: {config_history['mean_auc_pr']:.4f}")
        for fold_idx, fold_history in enumerate(config_history['history']):
            print(f"\nFold {fold_idx + 1}:")
            plot_training_history(fold_history)


def plot_confusion_matrix(y_true, y_pred, classes, title='Matriz de Confusión', cmap=plt.cm.Blues):
    """
    Calcula y grafica la matriz de confusión usando un heatmap.

    Parámetros:
      - y_true: Etiquetas verdaderas.
      - y_pred: Etiquetas predichas.
      - classes: Lista de nombres para cada clase.
      - title: Título del gráfico.
      - cmap: Mapa de colores.
    """
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Configurar el tamaño de la figura
    plt.figure(figsize=(6, 6))
    
    # Graficar con seaborn heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------------------------------------------------
# Elementos a guardar
# -------------------------------------------------------------------

import json
import datetime
import sys

# Calcular las predicciones y métricas finales:
y_pred_prob = final_model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= threshold).astype(int)

auc_pr = average_precision_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# -------------------------------------------------------------------
# 1. Guardar la configuración de hiperparámetros (best_config)
# -------------------------------------------------------------------
with open("hiperparametros.json", "w") as f:
    json.dump(best_config, f, indent=4)
print("La configuración de hiperparámetros se ha guardado en 'hiperparametros.json'.")

# -------------------------------------------------------------------
# 2. Guardar el modelo entrenado
# -------------------------------------------------------------------
# Puedes guardar el modelo en formato HDF5 o en el formato SavedModel de TF.
# Formato HDF5:
final_model.save("final_model.h5")
print("El modelo entrenado se ha guardado en 'final_model.h5'.")
#
# Alternativamente, para guardar en formato SavedModel:
# final_model.save("final_model", save_format="tf")

# -------------------------------------------------------------------
# 3. Guardar los resultados de evaluación (métricas)
# -------------------------------------------------------------------
resultados_evaluacion = {
    "auc_pr": auc_pr,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
with open("resultados_evaluacion.json", "w") as f:
    json.dump(resultados_evaluacion, f, indent=4)
print("Los resultados de evaluación se han guardado en 'resultados_evaluacion.json'.")

# -------------------------------------------------------------------
# 4. Guardar la matriz de confusión
# -------------------------------------------------------------------
# Convertimos la matriz (numpy array) a lista para que sea serializable en JSON.
with open("matriz_confusion.json", "w") as f:
    json.dump(cm.tolist(), f, indent=4)
print("La matriz de confusión se ha guardado en 'matriz_confusion.json'.")

# -------------------------------------------------------------------
# 5. Guardar el historial de entrenamiento (global_history)
# -------------------------------------------------------------------
with open("global_history.json", "w") as f:
    json.dump(global_history, f, indent=4)
print("El historial global se ha guardado en 'global_history.json'.")

# -------------------------------------------------------------------
# 6. Guardar metadatos del experimento (fecha, semilla, versiones, etc.)
# -------------------------------------------------------------------
metadatos_experimento = {
    "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "semilla": 42,
    "versiones": {
        "tensorflow": tf.__version__,
        "python": sys.version
    }
}
with open("metadatos_experimento.json", "w") as f:
    json.dump(metadatos_experimento, f, indent=4)
print("Los metadatos del experimento se han guardado en 'metadatos_experimento.json'.")
