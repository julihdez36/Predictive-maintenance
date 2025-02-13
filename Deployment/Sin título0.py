
"""
Utilizamos autoencoding para reducción de dimensionalidad y clasificamos sobre la representación latente.
El espacio de búsqueda se reduce solo a la arquitectura de la red (número de capas, unidades y activación).
"""

# --------------------------------------------
# 1. Importaciones
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, precision_recall_curve, f1_score, recall_score, precision_score
)
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition
from smac.facade import HyperbandFacade
from smac import Scenario
from imblearn.combine import SMOTETomek

# --------------------------------------------
# 2. Configuración Inicial y Semillas
# --------------------------------------------
tf.random.set_seed(42)
np.random.seed(42)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# --------------------------------------------
# 3. Carga y Preparación de Datos
# --------------------------------------------
def cargar_datos(url):
    df_final = pd.read_csv(url)
    X = df_final.drop(columns=['burned_transformers', 'year'])
    y = df_final['burned_transformers']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# URL de los datos
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_2.csv'
X_train_full, X_test, y_train_full, y_test = cargar_datos(url)

# --------------------------------------------
# 3.1. Función de Autoencoding para Reducción de Dimensionalidad
# --------------------------------------------
def autoencode_features(X_train, X_val, encoding_dim=20, epochs=50, batch_size=32):
    """
    Define y entrena un autoencoder simple para reducir la dimensionalidad.
    Retorna las versiones codificadas de X_train y X_val, además del autoencoder y el encoder.
    """
    input_dim = X_train.shape[1]
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
    encoder = tf.keras.models.Model(inputs=input_layer, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, X_val),
                    verbose=0)
    
    X_train_encoded = encoder.predict(X_train)
    X_val_encoded = encoder.predict(X_val)
    return X_train_encoded, X_val_encoded, autoencoder, encoder

# --------------------------------------------
# 4. Preprocesamiento de Datos con QuantileTransformer y Autoencoding
# --------------------------------------------
def preprocesar_datos(X_train_raw, y_train_raw, X_val_raw=None,
                      aplicar_autoencoder=False, encoding_dim=20,
                      ae_epochs=50, ae_batch_size=32, scaler=None):
    """
    - Escala los datos usando QuantileTransformer.
    - Aplica SMOTETomek para balancear las clases.
    - Si aplicar_autoencoder=True, entrena un autoencoder y transforma los datos a su representación latente.
    """
    # Ajustar el escalador (si no se proporciona)
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
    
    # Balancear clases
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    # Escalar datos
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    # Aplicar autoencoder para reducción de dimensionalidad (si se solicita)
    if aplicar_autoencoder and X_val_scaled is not None:
        X_train_final, X_val_final, autoencoder, encoder = autoencode_features(
            X_train_scaled, X_val_scaled,
            encoding_dim=encoding_dim,
            epochs=ae_epochs,
            batch_size=ae_batch_size
        )
    else:
        X_train_final, X_val_final = X_train_scaled, X_val_scaled
        encoder = None
    
    return X_train_final, y_train_bal, X_val_final, scaler, encoder

# --------------------------------------------
# 5. Función de Pérdida Focal para Datos Desbalanceados
# --------------------------------------------
def focal_loss(alpha=0.5, gamma=2.5):
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
# 6. Definición del Espacio de Búsqueda (solo arquitectura)
# --------------------------------------------
def get_configspace():
    """
    El espacio de búsqueda se reduce únicamente a:
      - Número de capas (num_layers)
      - Por cada capa: número de unidades y función de activación.
    """
    cs = ConfigurationSpace()
    max_layers = 12

    # Número de capas de la red
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    cs.add_hyperparameter(num_layers)
    
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
# 7. Arquitectura del Clasificador Basado en la Representación Latente
# --------------------------------------------
def create_model(best_config, input_dim):
    """
    Crea y compila el clasificador (sobre la representación latente) según la configuración.
    Solo se usan hiperparámetros relativos a la arquitectura.
    """
    model = Sequential()
    num_layers = best_config['num_layers']
    
    for i in range(num_layers):
        units = best_config[f"units_{i+1}"]
        activation = best_config[f"activation_{i+1}"]
        if i == 0:
            model.add(Dense(units, input_shape=(input_dim,)))
        else:
            model.add(Dense(units))
        model.add(Activation(activation))
    
    # Capa de salida para clasificación binaria
    model.add(Dense(1, activation='sigmoid'))
    
    # Parámetros fijos para el clasificador
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=focal_loss(alpha=0.5, gamma=2.5),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

# --------------------------------------------
# 8. Función para Buscar el Umbral Óptimo Basado en F1-score
# --------------------------------------------
def find_best_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

# --------------------------------------------
# 9. Función Objetivo para la Optimización de Hiperparámetros
# --------------------------------------------
global_history = []  # Para almacenar el historial de cada configuración

def objective_function(config, seed, budget):
    """
    Realiza validación con StratifiedKFold.
    Se preprocesan los datos (incluyendo reducción dimensional vía autoencoder) y se entrena el clasificador.
    """
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    fold_histories = []
    
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_raw, y_train_raw = X_train_full.iloc[train_idx], y_train_full.iloc[train_idx]
        X_val_raw, y_val_raw = X_train_full.iloc[val_idx], y_train_full.iloc[val_idx]
        
        # Preprocesamiento con autoencoding (reducción a representación latente)
        X_train_final, y_train_bal, X_val_final, scaler, encoder = preprocesar_datos(
            X_train_raw, y_train_raw, X_val_raw,
            aplicar_autoencoder=True,
            encoding_dim=20,    # Puedes ajustar este valor
            ae_epochs=50,
            ae_batch_size=32
        )
        
        model = create_model(config, input_dim=X_train_final.shape[1])
        
        # Calcular pesos de clases para compensar el desbalance
        class_weight = {
            0: len(y_train_bal) / (2 * sum(y_train_bal == 0)),
            1: len(y_train_bal) / (2 * sum(y_train_bal == 1))
        }
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=5, restore_best_weights=True)
        ]
        
        history = model.fit(
            X_train_final, y_train_bal,
            validation_data=(X_val_final, y_val_raw),
            epochs=int(budget),
            batch_size=32,
            verbose=0,
            class_weight=class_weight,
            callbacks=callbacks
        )
        
        # Guardar el historial de entrenamiento
        fold_histories.append({k: list(map(float, v)) for k, v in history.history.items()})
        
        y_val_pred = model.predict(X_val_final, verbose=0).ravel()
        auc_pr = average_precision_score(y_val_raw, y_val_pred)
        auc_pr_scores.append(auc_pr)
    
    global_history.append({
        "config": config,
        "history": fold_histories,
        "mean_auc_pr": np.mean(auc_pr_scores)
    })
    
    # Como SMAC/BOHB minimiza la función objetivo, se retorna 1 - AUC-PR promedio
    return 1 - np.mean(auc_pr_scores)

# --------------------------------------------
# 10. Optimización de Hiperparámetros con BOHB (SMAC)
# --------------------------------------------
def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=100,
        min_budget=5,
        max_budget=70,
        n_workers=4
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
# 11. Entrenamiento Final del Modelo con la Mejor Configuración
# --------------------------------------------
def entrenar_modelo(best_config, X_train_full, y_train_full):
    """
    Se separa un 10% de los datos para validación y se entrena el modelo final.
    Se aplica autoencoding para obtener la representación latente.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    # Aplicar autoencoder para obtener la representación latente
    X_train_final, X_val_final, autoencoder, encoder = autoencode_features(
        X_train_scaled, X_val_scaled,
        encoding_dim=20, epochs=50, batch_size=32
    )
    
    model = create_model(best_config, input_dim=X_train_final.shape[1])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=10, restore_best_weights=True, mode='max')
    ]
    
    model.fit(
        X_train_final, y_train_bal,
        validation_data=(X_val_final, y_val_split),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=callbacks
    )
    
    # Buscar el umbral óptimo basado en la validación
    y_val_pred = model.predict(X_val_final).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, best_threshold

# --------------------------------------------
# 12. Evaluación Final del Modelo en el Conjunto de Prueba
# --------------------------------------------
def evaluar_modelo(model, X_test, y_test, threshold):
    """
    Evalúa el modelo final e imprime las métricas, además de graficar la curva de precisión-recall.
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
    plt.title(f"Curva de Precisión-Recall (AUC-PR = {auc_pr:.4f})")
    plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------
# 13. Función para Graficar el Historial de Entrenamiento
# --------------------------------------------
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # AUC-PR
    plt.subplot(1, 2, 2)
    plt.plot(history['auc_pr'], label='Train AUC-PR')
    plt.plot(history['val_auc_pr'], label='Validation AUC-PR')
    plt.title('AUC-PR durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('AUC-PR')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("perdida_y_auc_pr.png", dpi=300, bbox_inches='tight')
    plt.show()

# --------------------------------------------
# 14. Función para Graficar la Matriz de Confusión
# --------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes, title='Matriz de Confusión', cmap=plt.cm.Blues):
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
# 15. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    # Fase 1: Optimización de Hiperparámetros con BOHB
    print("Iniciando optimización de hiperparámetros...")
    best_config = run_bohb_with_smac()
    print("Optimización completada. Mejor configuración encontrada:")
    print(best_config)
    
    # Fase 2: Entrenamiento final del modelo con la mejor configuración
    print("Entrenando el modelo final con la mejor configuración...")
    final_model, scaler, threshold = entrenar_modelo(best_config, X_train_full, y_train_full)
    
    # Transformar X_test: primero escalamos y luego aplicamos autoencoding
    X_test_scaled = scaler.transform(X_test)
    # Para transformar X_test usamos el mismo procedimiento del autoencoder (se entrena de forma independiente aquí)
    _, X_test_encoded, _, _ = autoencode_features(X_test_scaled, X_test_scaled, encoding_dim=20, epochs=50, batch_size=32)
    
    evaluar_modelo(final_model, X_test_encoded, y_test, threshold)
    
    # Visualizar historial de entrenamiento de cada configuración evaluada
    for i, config_history in enumerate(global_history):
        print(f"\nConfiguración {i + 1}:")
        print(f"Hiperparámetros: {config_history['config']}")
        print(f"AUC-PR promedio: {config_history['mean_auc_pr']:.4f}")
        for fold_idx, fold_history in enumerate(config_history['history']):
            print(f"\nFold {fold_idx + 1}:")
            plot_training_history(fold_history)
