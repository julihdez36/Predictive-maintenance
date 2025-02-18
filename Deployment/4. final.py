"""
Ajustes para mejorar la predicción de la clase minoritaria.

   1. Se reemplaza RobustScaler por QuantileTransformer con salida en distribución normal.
   2. Se usa StratifiedKFold en la función objetivo para la validación.
   3. Se aplica EarlyStopping basado en 'val_auc_pr' para evitar el sobreajuste de la clase mayoritaria.
   4. Se utiliza el callback OneCycleLR para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.
   5. La optimización se realiza con BOHB (mediante HyperbandFacade de SMAC) y se integra PCA (95% de varianza) para la extracción de características.
"""

# --------------------------------------------
# 1. Importaciones
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, precision_recall_curve, f1_score, recall_score, precision_score
)
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

# --------------------------------------------
# 2. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
# tf.config.threading.set_inter_op_parallelism_threads(10)
# tf.config.threading.set_intra_op_parallelism_threads(10)

# --------------------------------------------
# 3. Carga y Preparación de Datos
# --------------------------------------------

tf.random.set_seed(42)
np.random.seed(42)

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']

# Dividir en entrenamiento y prueba (para la optimización usaremos solo el entrenamiento)
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# --------------------------------------------
# 4. Preprocesamiento de Datos con QuantileTransformer y SMOTETomek
# --------------------------------------------
def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None, scaler=None):
    """
    Escala los datos usando QuantileTransformer para transformar la distribución
    a una distribución normal. Además, se aplica SMOTETomek para balancear el conjunto de entrenamiento.
    """
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
    
    # Aplicar SMOTETomek para balancear las clases
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    return X_train_scaled, y_train_bal, X_val_scaled, scaler

def preprocesar_datos_con_pca(X_train_scaled, y_train_raw, pca, aplicar_smote=False):
    """
    Transforma los datos escalados usando PCA y, opcionalmente,
    aplica SMOTETomek en el espacio reducido.
    """
    X_train_reducido = pca.transform(X_train_scaled)
    if aplicar_smote:
        smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
        X_train_reducido, y_train_bal = smote_tomek.fit_resample(X_train_reducido, y_train_raw)
    else:
        y_train_bal = y_train_raw
    return X_train_reducido, y_train_bal

def entrenar_pca(X_train_scaled, variance_threshold=0.95):
    """
    Ajusta PCA sobre los datos escalados conservando el porcentaje de varianza especificado.
    """
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(X_train_scaled)
    return pca

# --------------------------------------------
# 5. Callback OneCycleLR para Ajuste Dinámico de la Tasa de Aprendizaje
# --------------------------------------------
class OneCycleLR(tf.keras.callbacks.Callback):
    """
    Callback que implementa la política OneCycleLR.
    Durante el entrenamiento:
      - En la primera fase (~30% de las épocas), la tasa de aprendizaje aumenta linealmente.
      - En la segunda fase, disminuye linealmente hasta un valor final.
    """
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

# --------------------------------------------
# 6. Función de Pérdida Focal para Datos Desbalanceados
# --------------------------------------------
def focal_loss(alpha=0.95, gamma= 5):
    """
    Implementa la función de pérdida focal para enfocarse en ejemplos difíciles.
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
# 7. Arquitectura de la Red Neuronal (usando ConfigSpace)
# --------------------------------------------
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
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])
        if i > 1:
            cs.add_condition(InCondition(child=units, parent=num_layers, values=list(range(i, max_layers+1))))
            cs.add_condition(InCondition(child=activation, parent=num_layers, values=list(range(i, max_layers+1))))
    
    return cs

# --------------------------------------------
# 10. Función Objetivo para la Optimización de Hiperparámetros (usando PCA)
# --------------------------------------------
global_history = []  # Almacena los historiales de entrenamiento de cada configuración

def objective_function(config, seed, budget):
    """
    Función objetivo utilizada por SMAC/BOHB.
    Realiza validación con StratifiedKFold, integrando PCA (95% varianza) en el pipeline.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    fold_histories = []
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # Escalar los datos y aplicar balanceo (SMOTETomek) se hará después de PCA
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)
        
        # Ajustar PCA para conservar el 95% de la varianza
        pca = entrenar_pca(X_train_scaled, variance_threshold=0.95)
        
        # Obtener representación reducida y aplicar SMOTETomek (solo en entrenamiento)
        X_train_reducido, y_train_bal = preprocesar_datos_con_pca(X_train_scaled, y_train_raw, pca, aplicar_smote=True)
        # Para validación: solo se transforma
        X_val_reducido = pca.transform(X_val_scaled)
        
        input_dim = X_train_reducido.shape[1]
        model = create_model(config, input_dim=input_dim)
        
        # Cálculo de pesos de clases para compensar el desbalance
        class_weights = {0: 1, 1: 10}
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=5, restore_best_weights=True),
            OneCycleLR(max_lr=config['learning_rate'] * 10, epochs=int(budget))
        ]
        
        history = model.fit(
            X_train_reducido, y_train_bal,
            validation_data=(X_val_reducido, y_val_raw),
            epochs=int(budget),
            batch_size=config["batch_size"],
            verbose=0,
            callbacks=callbacks,
            class_weight= class_weights            
        )
        
        # Serializar el historial para guardarlo
        fold_histories.append({k: list(map(float, v)) for k, v in history.history.items()})
        
        y_val_pred = model.predict(X_val_reducido, verbose=0).ravel()
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)
    
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
# 12. Entrenamiento Final del Modelo con la Mejor Configuración (usando PCA)
# --------------------------------------------
def entrenar_modelo_con_pca(best_config, X_train_full, y_train_full, variance_threshold=0.95):
    """
    Flujo de entrenamiento final:
      - Se separa un 10% de los datos para validación.
      - Se escalan los datos.
      - Se ajusta PCA (95% de varianza) sobre los datos escalados.
      - Se aplica SMOTETomek en el espacio reducido.
      - Se entrena el clasificador sobre la representación reducida.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaler.fit(X_train_bal)
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    pca = entrenar_pca(X_train_scaled, variance_threshold)
    
    X_train_reducido, y_train_final = preprocesar_datos_con_pca(X_train_scaled, y_train_bal, pca, aplicar_smote=False)
    X_val_reducido = pca.transform(X_val_scaled)
    
    input_dim = X_train_reducido.shape[1]
    model = create_model(best_config, input_dim=input_dim)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
        OneCycleLR(max_lr=best_config['learning_rate'] * 10, epochs=100)
    ]
    
    model.fit(
        X_train_reducido, y_train_final,
        validation_data=(X_val_reducido, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks
    )
    
    y_val_pred = model.predict(X_val_reducido).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, pca, best_threshold

# --------------------------------------------
# 13. Evaluación Final del Modelo en el Conjunto de Prueba (usando PCA)
# --------------------------------------------
def evaluar_modelo_con_pca(model, pca, scaler, X_test, y_test, best_threshold):
    """
    Transforma X_test usando el scaler y PCA, y evalúa el modelo.
    """
    X_test_scaled = scaler.transform(X_test)
    X_test_reducido = pca.transform(X_test_scaled)
    y_pred_prob = model.predict(X_test_reducido).ravel()
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    
    auc_pr = average_precision_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.title(f"Curva de Precisión-Recall (AUC-PR = {auc_pr:.4f})")
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
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['auc_pr'], label='Train AUC-PR')
    plt.plot(history['val_auc_pr'], label='Validation AUC-PR')
    plt.title('AUC-PR durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('AUC-PR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("perdida_y_aucpr.png", dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------
# 15. Función para Graficar la Matriz de Confusión
# --------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, title='Matriz de Confusión', cmap=plt.cm.Blues):
    """
    Calcula y grafica la matriz de confusión usando un heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------
# 16. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    # Fase 1: Optimización de Hiperparámetros usando BOHB
    print("Iniciando optimización de hiperparámetros...")
    best_config = run_bohb_with_smac()
    print("Optimización completada. Mejor configuración encontrada:")
    print(best_config)

    # -------------------------------------------------------------------
    # Guardar Resultados y Artefactos del Experimento
    # -------------------------------------------------------------------
    # 1. Guardar la configuración de hiperparámetros
    best_config_dict = best_config.get_dictionary()
    with open("hiperparametros.json", "w") as f:
        json.dump(best_config_dict, f, indent=4)
    print("La configuración de hiperparámetros se ha guardado en 'hiperparametros.json'.")
