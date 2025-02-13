# Autoencoder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, classification_report, precision_score,
    recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve
)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from imblearn.combine import SMOTETomek
from tensorflow.keras.optimizers import Nadam

# --------------------------------------------
# 2. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)



def preprocesar_datos_con_autoencoder(X_train_raw, y_train_raw, encoder, aplicar_smote=False):
    """
    Preprocesa los datos aplicando el autoencoder y, opcionalmente, SMOTETomek.

    - X_train_raw: Datos crudos.
    - y_train_raw: Etiquetas.
    - encoder: Modelo encoder entrenado.
    - aplicar_smote: Indica si se aplica SMOTETomek después del autoencoder.
    """
    # Paso 1: Transformar los datos con el autoencoder
    X_train_encoded = encoder.predict(X_train_raw)
    
    # Paso 2: Aplicar SMOTETomek si es necesario
    if aplicar_smote:
        smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
        X_train_encoded, y_train_bal = smote_tomek.fit_resample(X_train_encoded, y_train_raw)
    else:
        y_train_bal = y_train_raw

    return X_train_encoded, y_train_bal


# --------------------------------------------
# Función para crear y entrenar el Autoencoder
# --------------------------------------------

def crear_autoencoder(input_dim, encoding_dim, regularizacion=0.001):
    """
    Crea un autoencoder simple con regularización L2.

    - input_dim: Dimensión de la entrada.
    - encoding_dim: Dimensión del espacio latente.
    - regularizacion: Factor de regularización L2.
    """
    input_layer = Input(shape=(input_dim,))

    # Capa de codificación con regularización L2
    encoded = Dense(encoding_dim, activation='relu', 
                  kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(input_layer)

    # Capa de decodificación con regularización L2
    decoded = Dense(input_dim, activation='sigmoid',
                  kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='Nadam', loss='mse')
    return autoencoder, encoder


# --------------------------------------------
# Función de Preprocesamiento (sin PCA para el autoencoder)
# --------------------------------------------
def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None,
                      aplicar_pca=False, n_componentes=0.95,
                      scaler=None, pca=None):
    """
    Preprocesa los datos de entrenamiento y validación aplicando:
      1. SMOTETomek para balancear las clases (sólo en entrenamiento).
      2. Escalado con QuantileTransformer.
      3. (Opcional) Reducción de dimensionalidad con PCA.
    
    En este caso, dejaremos aplicar_pca=False porque usaremos un autoencoder.
    """
    # 1. Ajustar el escalador en los datos de entrenamiento sin balancear
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)

    # 2. Aplicar SMOTETomek para balancear el conjunto de entrenamiento
    smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)

    # 3. Escalar los datos balanceados
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None

    # Si se aplicara PCA (no en este caso)
    if aplicar_pca:
        if pca is None:
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
# Función para crear y entrenar el Autoencoder
# --------------------------------------------
# def crear_autoencoder(input_dim, encoding_dim):
#     """
#     Crea un autoencoder simple.
#       - input_dim: Dimensión de la entrada.
#       - encoding_dim: Dimensión del espacio latente.
#     """
#     input_layer = Input(shape=(input_dim,))
#     # Capa de codificación
#     encoded = Dense(encoding_dim, activation='relu')(input_layer)
#     # Capa de decodificación
#     decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)
    
#     autoencoder.compile(optimizer='Nadam', loss='mse')
#     return autoencoder, encoder

def entrenar_autoencoder(X_train, X_val, encoding_dim, epochs=50, batch_size=32):
    """
    Entrena el autoencoder y devuelve el modelo autoencoder y su encoder.
    """
    input_dim = X_train.shape[1]
    autoencoder, encoder = crear_autoencoder(input_dim, encoding_dim)
    
    autoencoder.fit(X_train, X_train,
                    validation_data=(X_val, X_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1)
    return autoencoder, encoder

# --------------------------------------------
# (Funciones existentes: OneCycleLR, focal_loss, create_model, find_best_threshold, etc.)
# --------------------------------------------
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

def focal_loss(alpha=0.95, gamma=5):
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

def create_model(best_config, input_dim):
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

# --------------------------------------------
# Entrenamiento del Clasificador sobre Representación Latente
# --------------------------------------------
def entrenar_modelo_con_autoencoder(best_config, X_train_full, y_train_full, encoding_dim=64):
    """
    - Separa un 10% para validación.
    - Preprocesa los datos (SMOTETomek y escalado) sin PCA.
    - Entrena un autoencoder para obtener representaciones latentes.
    - Transforma los datos (entrenamiento, validación y prueba) usando el encoder.
    - Entrena el clasificador sobre la representación latente.
    """
    # División en entrenamiento y validación
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    # Preprocesamiento (sin PCA)
    X_train_proc, y_train_bal, X_val_proc, scaler, _ = preprocesar_datos(
        X_train_split, y_train_split, X_val_raw=X_val_split,
        aplicar_pca=False
    )
    
    # Entrenar el autoencoder para obtener representación latente
    autoencoder, encoder = entrenar_autoencoder(X_train_proc, X_val_proc, encoding_dim,
                                                 epochs=50, batch_size=32)
    
    # Extraer las representaciones latentes
    X_train_encoded = encoder.predict(X_train_proc)
    X_val_encoded   = encoder.predict(X_val_proc)
    
    # Crear el clasificador usando la dimensión latente
    model = create_model(best_config, input_dim=encoding_dim)
    
    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
        OneCycleLR(max_lr=best_config['learning_rate'] * 10, epochs=100)
    ]
    
    class_weights = {0: 1, 1: 10}
    # (Opcional) Puedes incluir class_weight si lo deseas
    model.fit(
        X_train_encoded, y_train_bal,
        validation_data=(X_val_encoded, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Buscar umbral óptimo en validación
    y_val_pred = model.predict(X_val_encoded).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, encoder, best_threshold

# --------------------------------------------
# Evaluación Final
# --------------------------------------------
def evaluar_modelo_con_autoencoder(model, encoder, scaler, X_test, y_test, best_threshold):
    """
    Transforma X_test usando el scaler y el encoder, y evalúa el modelo.
    """
    X_test_scaled = scaler.transform(X_test)
    X_test_encoded = encoder.predict(X_test_scaled)
    y_pred_prob = model.predict(X_test_encoded).ravel()
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
# Pipeline Completo de Entrenamiento y Evaluación
# --------------------------------------------

# Cargar datos
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_2.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['burned_transformers'])
y = df_final['burned_transformers']

# Dividir en entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Definir hiperparámetros para el clasificador (puedes ajustar estos valores)
hiperparametrospca = {
    'activation_1': 'swish',
    'batch_size': 32, #probar cambios
    'dropout': 0.2561936143454,
    'learning_rate': 0.0013331697203, # probas cambios
    'num_layers': 7,
    'optimizer': 'Nadam', # 'adam'
    'units_1': 148,
    'use_l1': True,
    'use_l2': True,
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

# Define la dimensión del espacio latente del autoencoder
encoding_dim = 64

# Entrenar el clasificador usando el autoencoder para extraer representaciones latentes
model, scaler, encoder, best_threshold = entrenar_modelo_con_autoencoder(hiperparametrospca, X_train_full, y_train_full, encoding_dim)

# Evaluar en el conjunto de prueba
evaluar_modelo_con_autoencoder(model, encoder, scaler, X_test, y_test, best_threshold)




##### Con Leaky RELU ##########################

# def crear_autoencoder(input_dim, encoding_dim, regularizacion=0.001):
#     """
#     Crea un autoencoder con LeakyReLU y regularización L2.
    
#     - input_dim: Dimensión de entrada.
#     - encoding_dim: Dimensión del espacio latente.
#     - regularizacion: Factor de regularización L2.
#     """
#     input_layer = Input(shape=(input_dim,))

#     # Codificación con LeakyReLU y regularización
#     encoded = Dense(encoding_dim, kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(input_layer)
#     encoded = LeakyReLU(alpha=0.1)(encoded)
    
    
#     # Decodificación con LeakyReLU
#     decoded = Dense(input_dim, kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(encoded)
#     decoded = LeakyReLU(alpha=0.1)(decoded)

#     # Modelo Autoencoder
#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)

#     autoencoder.compile(optimizer='Nadam', loss=tf.keras.losses.LogCosh())  # loss='mse' también es una opción
#     return autoencoder, encoder

# def crear_autoencoder(input_dim, encoding_dim, regularizacion=0.001):
#     input_layer = Input(shape=(input_dim,))

#     # Capas más profundas
#     encoded = Dense(128, activation='relu')(input_layer)
#     encoded = Dense(64, activation='relu')(encoded)
#     encoded = Dense(encoding_dim, activation='relu', 
#                     kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(encoded)

#     decoded = Dense(64, activation='relu')(encoded)
#     decoded = Dense(128, activation='relu')(decoded)
#     decoded = Dense(input_dim, activation='sigmoid')(decoded)

#     autoencoder = Model(input_layer, decoded)
#     encoder = Model(input_layer, encoded)

#     autoencoder.compile(optimizer='Nadam', loss=tf.keras.losses.LogCosh())
#     return autoencoder, encoder

