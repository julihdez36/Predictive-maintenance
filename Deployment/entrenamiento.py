"""
Modelo cargado con PCA

"""

# --------------------------------------------
# 1. Importaciones
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split#, StratifiedKFold
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

from imblearn.combine import SMOTETomek
from sklearn.decomposition import PCA


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
# 3.1. PCA
# --------------------------------------------


def reducir_dimensionalidad(X_train, X_test, n_componentes=0.95):
    pca = PCA(n_components=n_componentes)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca

# --------------------------------------------
# 4. Preprocesamiento de Datos con QuantileTransformer
# --------------------------------------------


def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None, aplicar_pca=True, n_componentes=0.95, scaler=None, pca=None):
    """
    Escala los datos usando QuantileTransformer y aplica SMOTETomek para balancear el conjunto de entrenamiento.
    Opcionalmente, aplica PCA para reducción de dimensionalidad.
    
    Parámetros:
        - X_train_raw: Datos de entrenamiento (sin procesar).
        - y_train_raw: Etiquetas de entrenamiento.
        - X_val_raw: Datos de validación (opcional).
        - aplicar_pca: Si True, aplica PCA.
        - n_componentes: Número de componentes principales a retener en PCA.
        - scaler: Objeto QuantileTransformer preajustado (opcional).
        - pca: Objeto PCA preajustado (opcional).
    
    Retorna:
        - X_train_final: Datos de entrenamiento preprocesados.
        - y_train_bal: Etiquetas de entrenamiento balanceadas.
        - X_val_final: Datos de validación preprocesados (o None si no se proporciona).
        - scaler: Objeto QuantileTransformer ajustado.
        - pca: Objeto PCA ajustado (o None si no se usa PCA).
    """
    
    # Inicializar y ajustar el escalador si no se proporciona
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)

    # Aplicar SMOTETomek para balancear las clases
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)

    # Escalar datos
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None

    # Aplicar PCA si es necesario
    if aplicar_pca:
        if pca is None:  # Ajustar PCA solo en los datos de entrenamiento
            pca = PCA(n_components=n_componentes)
            pca.fit(X_train_scaled)

        X_train_final = pca.transform(X_train_scaled)
        X_val_final = pca.transform(X_val_scaled) if X_val_scaled is not None else None
    else:
        X_train_final = X_train_scaled
        X_val_final = X_val_scaled
        pca = None

    return X_train_final, y_train_bal, X_val_final, scaler, pca

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

# -------------------------------------------------------------------
# Fase 2: Entrenamiento final del modelo con la mejor configuración
# -------------------------------------------------------------------

hiperparametros = {
  'activation_1': 'swish',
  'batch_size': 32,
  'dropout': 0.2561936143454,
  'learning_rate': 0.0013331697203,
  'num_layers': 7,
  'optimizer': 'adam',
  'units_1': 148,
  'use_l1': False,
  'use_l2': False,
  'activation_2': 'swish',
  'activation_3': 'tanh',
  'activation_4': 'swish',
  'activation_5': 'swish',
  'activation_6': 'elu',
  'activation_7': 'tanh',
  'units_2': 171,
  'units_3': 181,
  'units_4': 119,
  'units_5': 142,
  'units_6': 128,
  'units_7': 233,
}

# Sin PCA
print("Entrenando el modelo final con la mejor configuración...")
final_model, scaler, threshold = entrenar_modelo(hiperparametros, X_train_full, y_train_full)
    
# Evaluación en el conjunto de prueba
X_test_scaled = scaler.transform(X_test)
evaluar_modelo(final_model, X_test_scaled, y_test, threshold)

    
# Calcular las predicciones y métricas finales:
y_pred_prob = final_model.predict(X_test_scaled).ravel()
y_pred = (y_pred_prob >= threshold).astype(int)

auc_pr = average_precision_score(y_test, y_pred_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


# Con PCA



X_train_pca, X_test_pca, pca = reducir_dimensionalidad(X_train_full, X_test)

print("Entrenando el modelo final con la mejor configuración...")
final_modelpca, scalerpca, thresholdpca = entrenar_modelo(hiperparametros, X_train_pca, y_train_full)

X_test_scaled = scalerpca.transform(X_test)
    

