# --------------------------------------------
# Autoencoder y Clasificador sobre Representación Latente
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, classification_report, precision_score,
    recall_score, f1_score, PrecisionRecallDisplay, precision_recall_curve, confusion_matrix
)

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam
from imblearn.combine import SMOTETomek

# --------------------------------------------
# Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)

# tf.config.threading.set_inter_op_parallelism_threads(4)
# tf.config.threading.set_intra_op_parallelism_threads(4)

# --------------------------------------------
# Función para transformar datos usando el encoder y aplicar SMOTETomek en el espacio latente
# --------------------------------------------
def preprocesar_datos_con_autoencoder(X_train_raw, y_train_raw, encoder, aplicar_smote=False):
    """
    Transforma los datos de entrada usando el encoder entrenado y, opcionalmente,
    aplica SMOTETomek en el espacio latente.

    - X_train_raw: Datos de entrada escalados.
    - y_train_raw: Etiquetas originales.
    - encoder: Modelo encoder entrenado.
    - aplicar_smote: Si True, aplica SMOTETomek para balancear las clases.
    """
    # Transformar los datos a la representación latente
    X_train_encoded = encoder.predict(X_train_raw)
    
    # Aplicar SMOTETomek en el espacio latente, si se requiere
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

    # Capa de codificación con regularización L2 LEAKY
    encoded = Dense(encoding_dim, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(input_layer)
  
    decoded = Dense(input_dim, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='nadam', loss= tf.keras.losses.LogCosh()) #loss='mse'
    return autoencoder, encoder


##### Entrenar modelo #################333

def entrenar_autoencoder(X_train, X_val, encoding_dim, epochs=100, batch_size=32):
    """
    Entrena el autoencoder con los datos escalados y devuelve el modelo entrenado.
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
# Callbacks y funciones auxiliares para el clasificador
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

# --------------------------------------------
# Entrenamiento del Clasificador sobre Representación Latente
# --------------------------------------------
def entrenar_modelo_con_autoencoder(best_config, X_train_full, y_train_full, encoding_dim=64):
    """
    Flujo de entrenamiento:
      - Se separa un 10% para validación.
      - Se escalan los datos sin balancear.
      - Se entrena el autoencoder sobre los datos escalados.
      - Se transforma la representación latente y se aplica SMOTETomek en el espacio latente (opcional).
      - Se entrena el clasificador sobre la representación latente balanceada.
    """
    # División en entrenamiento y validación
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    # Escalar los datos sin aplicar SMOTETomek previamente
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    scaler.fit(X_train_split)
    X_train_scaled = scaler.transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    
    # Entrenar el autoencoder en datos escalados
    autoencoder, encoder = entrenar_autoencoder(X_train_scaled, X_val_scaled, encoding_dim,
                                                 epochs=50, batch_size=32)
    
    # Obtener la representación latente y aplicar SMOTETomek en el espacio latente (solo en entrenamiento)
    X_train_encoded, y_train_final = preprocesar_datos_con_autoencoder(X_train_scaled, y_train_split, encoder, aplicar_smote=True)
    
    # Para validación, usar la transformación del encoder sin SMOTETomek
    X_val_encoded = encoder.predict(X_val_scaled)
    
    # Crear el clasificador usando la dimensión latente
    model = create_model(best_config, input_dim=encoding_dim)
    
    # Configurar callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
        OneCycleLR(max_lr=best_config['learning_rate'] * 10, epochs=100)
    ]
    
    class_weights = {0: 1, 1: 10}
    model.fit(
        X_train_encoded, y_train_final,
        validation_data=(X_val_encoded, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Buscar el umbral óptimo en validación
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
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']

X.isna().sum()

# Dividir en entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Definir hiperparámetros para el clasificador (ajustar según sea necesario)

# hiperparametrospca = {
#     'activation_1': 'swish',
#     'batch_size': 32, #probar cambios
#     'dropout': 0.2561936143454,
#     'learning_rate': 0.0013331697203, # probas cambios
#     'num_layers': 7,
#     'optimizer': 'Nadam', # 'adam'
#     'units_1': 148,
#     'use_l1': True,
#     'use_l2': True,
#     'activation_2': 'swish',
#     'activation_3': 'tanh',
#     'activation_4': 'swish',
#     'activation_5': 'swish',
#     'activation_6': 'elu',
#     'activation_7': 'tanh',
#     'units_2': 171,
#     'units_3': 181,
#     'units_4': 119,
#     'units_5': 142,
#     'units_6': 128,
#     'units_7': 233,
# }

hiperparametros = {
    "activation_1": "tanh",
    "batch_size": 32,
    "dropout": 0.0540769997851,
    "learning_rate": 0.0012889055447,
    "num_layers": 3,
    "optimizer": "rmsprop", 
    "units_1": 11,
    "use_l1": True,
    "use_l2": True,
    "activation_2": "swish",
    "activation_3": "relu",
    "units_2": 225,
    "units_3": 27
  }


# Definir la dimensión del espacio latente del autoencoder
encoding_dim = 12 #es el que mejor rendimiento ha mostrado

# Entrenar el clasificador usando el autoencoder para extraer representaciones latentes
model, scaler, encoder, best_threshold = entrenar_modelo_con_autoencoder(hiperparametros, X_train_full, y_train_full, encoding_dim)

# Evaluar en el conjunto de prueba
evaluar_modelo_con_autoencoder(model, encoder, scaler, X_test, y_test, best_threshold)
best_threshold

# Utiliza el encoder y el modelo para obtener las predicciones en el conjunto de prueba
predicciones_latentes = encoder.predict(scaler.transform(X_test))
predicciones_probabilidades = model.predict(predicciones_latentes)
predicciones_etiquetas = (predicciones_probabilidades > best_threshold).astype(int)  # Ajusta el umbral si es necesario


matriz_confusion = confusion_matrix(y_test, predicciones_etiquetas); matriz_confusion



