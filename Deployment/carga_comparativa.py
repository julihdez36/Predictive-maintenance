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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
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

# --------------------------------------------
# 3.1. PCA
# --------------------------------------------


def reducir_dimensionalidad(X_train, X_test, n_componentes=0.95):
    pca = PCA(n_components=n_componentes)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca, pca


# --- Función de Preprocesamiento con SMOTETomek, QuantileTransformer y PCA ---

def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None,
                      aplicar_pca=True, n_componentes=0.95,
                      scaler=None, pca=None):
    """
    Preprocesa los datos de entrenamiento y validación aplicando:
      1. SMOTETomek para balancear las clases (sólo en entrenamiento).
      2. Escalado con QuantileTransformer.
      3. Reducción de dimensionalidad con PCA (opcional).
    
    Parámetros:
        - X_train_raw: Datos de entrenamiento sin procesar.
        - y_train_raw: Etiquetas de entrenamiento.
        - X_val_raw: Datos de validación sin procesar (opcional).
        - aplicar_pca: Si True, aplica PCA.
        - n_componentes: Porcentaje o número de componentes a retener en PCA.
        - scaler: Objeto QuantileTransformer preajustado (opcional).
        - pca: Objeto PCA preajustado (opcional).
    
    Retorna:
        - X_train_final: Datos de entrenamiento preprocesados.
        - y_train_bal: Etiquetas de entrenamiento balanceadas.
        - X_val_final: Datos de validación preprocesados (None si no se proporciona X_val_raw).
        - scaler: Objeto QuantileTransformer ajustado.
        - pca: Objeto PCA ajustado (None si aplicar_pca es False).
    """
    # 1. Ajustar el escalador en los datos de entrenamiento sin balancear
    if scaler is None:
        #scaler = RobustScaler()
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)

    # 2. Aplicar SMOTETomek para balancear el conjunto de entrenamiento
    smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)

    # 3. Escalar los datos balanceados
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None

    # 4. Aplicar PCA (ajustado únicamente en el conjunto de entrenamiento)
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
def focal_loss(alpha=0.95, gamma=4):
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

# def find_best_threshold(y_true, y_pred, min_precision=0.5):
#     """
#     Encuentra el umbral que maximiza el recall, asegurando una precisión mínima.
#     """
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    
#     # Asegúrate de que las dimensiones coincidan
#     precisions = precisions[:-1]  # Elimina el último valor de precisions y recalls
#     recalls = recalls[:-1]
    
#     # Filtra los umbrales que cumplen con la precisión mínima
#     viable_thresholds = thresholds[precisions >= min_precision]
    
#     # Si no hay umbrales viables, devuelve el umbral por defecto (0.5)
#     if len(viable_thresholds) == 0:
#         return 0.5
    
#     # Devuelve el umbral que maximiza el recall
#     return viable_thresholds[np.argmax(recalls[precisions >= min_precision])]

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
    smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    #scaler = RobustScaler()
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


# --- Pipeline Completo de Entrenamiento y Evaluación ---

# Supongamos que ya tienes cargados tus datos:
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_2.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['burned_transformers'])
y = df_final['burned_transformers']

# Dividir en entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Separar una parte del entrenamiento para validación
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
)

# Preprocesar los datos: se aplica SMOTETomek, escalado y PCA en el conjunto de entrenamiento
X_train_final, y_train_bal, X_val_final, scaler, pca = preprocesar_datos(
    X_train_split, y_train_split, X_val_raw=X_val_split,
    aplicar_pca=True, n_componentes=0.95
)

# Definir los hiperparámetros 

# Hiperparámetros modelo PCA
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


# Hiperparámetros modelo 1
hiperparametros1 = {
    "activation_1": "tanh",
    "batch_size": 32,
    "dropout": 0.1605529295493,
    "learning_rate": 0.0028080828781,
    "num_layers": 3,
    "optimizer": "rmsprop",
    "units_1": 36,
    "use_l1": False,
    "use_l2": True,
    "activation_2": "relu",
    "activation_3": "tanh",
    "l2": 2.26379781e-05,
    "units_2": 206,
    "units_3": 109
}

# Hiperpaámetros modelo 2

hiperparametros2 = {
  'activation_1': 'elu',
  'batch_size': 64,
  'dropout': 0.2338718221271,
  'learning_rate': 8.06492543e-05,
  'num_layers': 6,
  'optimizer': 'rmsprop',
  'units_1': 56,
  'use_l1': False,
  'use_l2': True,
  'activation_2': 'swish',
  'activation_3': 'relu',
  'activation_4': 'swish',
  'activation_5': 'tanh',
  'activation_6': 'swish',
  'l2': 0.0008443519724,
  'units_2': 120,
  'units_3': 224,
  'units_4': 11,
  'units_5': 11,
  'units_6': 14,
}


# --------------------------------------------
# Aquitectura PCA
# --------------------------------------------

# Crear el modelo utilizando la dimensionalidad resultante tras PCA
modelpca = create_model(hiperparametrospca, input_dim=X_train_final.shape[1])

# Configurar callbacks, incluyendo OneCycleLR y EarlyStopping (asegúrate de tener definida la clase OneCycleLR)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
    OneCycleLR(max_lr=hiperparametrospca['learning_rate'] * 10, epochs=100)
]

# Entrenar el modelo
history = modelpca.fit(
    X_train_final, y_train_bal,
    validation_data=(X_val_final, y_val_split),
    epochs=100,
    batch_size=hiperparametrospca["batch_size"],
    verbose=1,
    callbacks=callbacks
)

# Encontrar el umbral óptimo basado en el conjunto de validación
y_val_pred = modelpca.predict(X_val_final).ravel()
best_threshold = find_best_threshold(y_val_split, y_val_pred)

# --- Evaluación Final en el Conjunto de Prueba ---

# Recordemos que el conjunto de prueba NO se balancea con SMOTETomek;
# solo se transforma usando el mismo scaler y PCA ajustados.
X_test_scaled = scaler.transform(X_test)
X_test_final = pca.transform(X_test_scaled)

evaluar_modelo(modelpca, X_test_final, y_test, best_threshold)

# --------------------------------------------
# Aquitectura 1
# --------------------------------------------

# Crear el modelo utilizando la dimensionalidad resultante tras PCA
model1 = create_model(hiperparametros1, input_dim=X_train_final.shape[1])

# Configurar callbacks, incluyendo OneCycleLR y EarlyStopping (asegúrate de tener definida la clase OneCycleLR)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
    OneCycleLR(max_lr=hiperparametrospca['learning_rate'] * 10, epochs=100)
]

# Entrenar el modelo
history = model1.fit(
    X_train_final, y_train_bal,
    validation_data=(X_val_final, y_val_split),
    epochs=100,
    batch_size=hiperparametrospca["batch_size"],
    verbose=1,
    callbacks=callbacks
)

# Encontrar el umbral óptimo basado en el conjunto de validación
y_val_pred = model1.predict(X_val_final).ravel()
best_threshold = find_best_threshold(y_val_split, y_val_pred)

# --- Evaluación Final en el Conjunto de Prueba ---

# Recordemos que el conjunto de prueba NO se balancea con SMOTETomek;
# solo se transforma usando el mismo scaler y PCA ajustados.
X_test_scaled = scaler.transform(X_test)
X_test_final = pca.transform(X_test_scaled)

evaluar_modelo(model1, X_test_final, y_test, .6)
# --------------------------------------------
# Aquitectura 2
# --------------------------------------------

# Crear el modelo utilizando la dimensionalidad resultante tras PCA
model2 = create_model(hiperparametros2, input_dim=X_train_final.shape[1])

# Configurar callbacks, incluyendo OneCycleLR y EarlyStopping (asegúrate de tener definida la clase OneCycleLR)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
    OneCycleLR(max_lr=hiperparametrospca['learning_rate'] * 10, epochs=100)
]

# Entrenar el modelo
history = model2.fit(
    X_train_final, y_train_bal,
    validation_data=(X_val_final, y_val_split),
    epochs=100,
    batch_size=hiperparametrospca["batch_size"],
    verbose=1,
    callbacks=callbacks
)

# Encontrar el umbral óptimo basado en el conjunto de validación
y_val_pred = model2.predict(X_val_final).ravel()
best_threshold = find_best_threshold(y_val_split, y_val_pred)

# --- Evaluación Final en el Conjunto de Prueba ---

# Recordemos que el conjunto de prueba NO se balancea con SMOTETomek;
# solo se transforma usando el mismo scaler y PCA ajustados.
X_test_scaled = scaler.transform(X_test)
X_test_final = pca.transform(X_test_scaled)

evaluar_modelo(model2, X_test_final, y_test, best_threshold)


plt.hist(y_val_pred, bins=50)
plt.title("Distribución de probabilidades predichas")
plt.show()


# --------------------------------------------
# ADASYN (para sobremuestrear la clase minoritaria) y RandomUnderSampler (para ajustar la cantidad de la clase mayoritaria)
# --------------------------------------------

def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None,
                      aplicar_pca=True, n_componentes=0.95,
                      scaler=None, pca=None):
    """
    Preprocesa los datos de entrenamiento y validación aplicando:
      1. Balanceo de clases con ADASYN y RandomUnderSampler (sólo en entrenamiento).
      2. Escalado con QuantileTransformer (u otro escalador si se prefiere).
      3. Reducción de dimensionalidad con PCA (opcional).
    
    Parámetros:
        - X_train_raw: Datos de entrenamiento sin procesar.
        - y_train_raw: Etiquetas de entrenamiento.
        - X_val_raw: Datos de validación sin procesar (opcional).
        - aplicar_pca: Si True, aplica PCA.
        - n_componentes: Porcentaje o número de componentes a retener en PCA.
        - scaler: Objeto QuantileTransformer preajustado (opcional).
        - pca: Objeto PCA preajustado (opcional).
    
    Retorna:
        - X_train_final: Datos de entrenamiento preprocesados.
        - y_train_bal: Etiquetas de entrenamiento balanceadas.
        - X_val_final: Datos de validación preprocesados (None si no se proporciona X_val_raw).
        - scaler: Objeto QuantileTransformer ajustado.
        - pca: Objeto PCA ajustado (None si aplicar_pca es False).
    """
    # 1. Ajustar el escalador en los datos de entrenamiento sin balancear
    if scaler is None:
        # Puedes cambiar a RobustScaler u otro escalador si lo prefieres
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)

    # 2. Balancear el conjunto de entrenamiento usando ADASYN y RandomUnderSampler
    from imblearn.over_sampling import ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    # Define el sobremuestreo y el submuestreo; ajusta los ratios según convenga
    over = ADASYN(sampling_strategy=0.8, random_state=42)
    under = RandomUnderSampler(sampling_strategy=1, random_state=42)
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X_train_bal, y_train_bal = pipeline.fit_resample(X_train_raw, y_train_raw)

    # 3. Escalar los datos balanceados
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None

    # 4. Aplicar PCA (ajustado únicamente en el conjunto de entrenamiento)
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


def entrenar_modelo(best_config, X_train_full, y_train_full):
    """
    Entrena el modelo final usando la mejor configuración.
    Se separa un 10% de los datos para validación (usando train_test_split) para monitorear el EarlyStopping.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    # Balancear el conjunto de entrenamiento con ADASYN y RandomUnderSampler
    from imblearn.over_sampling import ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    over = ADASYN(sampling_strategy=1, random_state=42)
    under = RandomUnderSampler(sampling_strategy=1, random_state=42)
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X_train_bal, y_train_bal = pipeline.fit_resample(X_train_split, y_train_split)
    
    # Escalado: puedes cambiar a otro escalador si lo deseas
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


# Supongamos que ya tienes cargados tus datos:
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_2.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['burned_transformers'])
y = df_final['burned_transformers']

# Dividir en entrenamiento y prueba
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Separar una parte del entrenamiento para validación
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
)

# Preprocesar los datos: se aplica ADASYN, escalado y PCA en el conjunto de entrenamiento
X_train_final, y_train_bal, X_val_final, scaler, pca = preprocesar_datos(
    X_train_split, y_train_split, X_val_raw=X_val_split,
    aplicar_pca=True, n_componentes=0.95
)


# Crear el modelo utilizando la dimensionalidad resultante tras PCA
modelpca = create_model(hiperparametrospca, input_dim=X_train_final.shape[1])

# Configurar callbacks, incluyendo OneCycleLR y EarlyStopping (asegúrate de tener definida la clase OneCycleLR)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max'),
    OneCycleLR(max_lr=hiperparametrospca['learning_rate'] * 10, epochs=100)
]

# Entrenar el modelo
history = modelpca.fit(
    X_train_final, y_train_bal,
    validation_data=(X_val_final, y_val_split),
    epochs=100,
    batch_size=hiperparametrospca["batch_size"],
    verbose=1,
    callbacks=callbacks
)

# Encontrar el umbral óptimo basado en el conjunto de validación
y_val_pred = modelpca.predict(X_val_final).ravel()
best_threshold = find_best_threshold(y_val_split, y_val_pred)

# --- Evaluación Final en el Conjunto de Prueba ---

# Recordemos que el conjunto de prueba NO se balancea con SMOTETomek;
# solo se transforma usando el mismo scaler y PCA ajustados.
X_test_scaled = scaler.transform(X_test)
X_test_final = pca.transform(X_test_scaled)

evaluar_modelo(modelpca, X_test_final, y_test, best_threshold)

# --------------------------------------------
# Probando con folds
# --------------------------------------------

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score
import numpy as np

# Configurar K-Fold
n_splits = 5  # Número de folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Listas para almacenar métricas y modelos
auc_pr_scores = []
recall_scores = []
models = []

# Iterar sobre los folds
for fold, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full)):
    print(f"\nEntrenando fold {fold + 1}/{n_splits}")
    
    # Dividir los datos en entrenamiento y validación para este fold
    X_train_fold, X_val_fold = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
    y_train_fold, y_val_fold = y_train_full.iloc[train_index], y_train_full.iloc[val_index]
    
    # Preprocesar los datos (SMOTETomek, escalado, PCA)
    X_train_final, y_train_bal, X_val_final, scaler, pca = preprocesar_datos(
        X_train_fold, y_train_fold, X_val_raw=X_val_fold,
        aplicar_pca=True, n_componentes=0.95
    )
    
    # Crear y entrenar el modelo con los mismos hiperparámetros
    model = create_model(hiperparametrospca, input_dim=X_train_final.shape[1])
    history = model.fit(
        X_train_final, y_train_bal,
        validation_data=(X_val_final, y_val_fold),
        epochs=100,
        batch_size=hiperparametrospca["batch_size"],
        verbose=1,
        callbacks=callbacks
    )
    
    # Encontrar el umbral óptimo para este fold
    y_val_pred = model.predict(X_val_final).ravel()
    best_threshold = find_best_threshold(y_val_fold, y_val_pred, min_precision=0.5)
    
    # Evaluar el modelo en el conjunto de validación
    y_val_pred_binary = (y_val_pred >= best_threshold).astype(int)
    auc_pr = average_precision_score(y_val_fold, y_val_pred)
    recall = recall_score(y_val_fold, y_val_pred_binary)
    
    # Almacenar métricas y modelo
    auc_pr_scores.append(auc_pr)
    recall_scores.append(recall)
    models.append(model)
    
    print(f"Fold {fold + 1} - AUC-PR: {auc_pr:.4f}, Recall: {recall:.4f}")

# Calcular métricas promedio
mean_auc_pr = np.mean(auc_pr_scores)
mean_recall = np.mean(recall_scores)
print(f"\nMétricas promedio - AUC-PR: {mean_auc_pr:.4f}, Recall: {mean_recall:.4f}")



