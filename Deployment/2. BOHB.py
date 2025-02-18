# --------------------------------------------
# 1. Importaciones
# --------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, precision_recall_curve, f1_score, recall_score, precision_score
)

from imblearn.combine import SMOTETomek

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from smac.facade import HyperbandFacade
from smac import Scenario

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
    Se asume que la columna a predecir es 'burned_transformers'.
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
    Escala los datos usando QuantileTransformer y aplica SMOTETomek para balancear.
    """
    if scaler is None:
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        scaler.fit(X_train_raw)
    
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_raw, y_train_raw)
    
    X_train_scaled = scaler.transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_raw) if X_val_raw is not None else None
    
    return X_train_scaled, y_train_bal, X_val_scaled, scaler

# --------------------------------------------
# 5. Callback OneCycleLR para Ajuste Dinámico de la Tasa de Aprendizaje
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

# --------------------------------------------
# 6. Función de Pérdida Focal para Datos Desbalanceados
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
# 7. Definición del Autoencoder Simple
# --------------------------------------------
def crear_autoencoder(input_dim, encoding_dim, regularizacion=0.001):
    """
    Crea un autoencoder simple con regularización L2.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid',
                    kernel_regularizer=tf.keras.regularizers.l2(regularizacion))(encoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    autoencoder.compile(optimizer='nadam', loss=tf.keras.losses.LogCosh())
    return autoencoder, encoder

def create_autoencoder_simple(config, input_dim):
    """
    Crea el autoencoder simple usando los hiperparámetros del espacio de búsqueda.
    """
    encoding_dim = config['ae_latent_dim']
    regularizacion = float(config['ae_regularizacion'])
    return crear_autoencoder(input_dim, encoding_dim, regularizacion)

# --------------------------------------------
# 8. Definición del Clasificador
# --------------------------------------------
def create_classifier(config, input_dim):
    """
    Crea y compila el clasificador basado en la configuración de hiperparámetros.
    """
    model = Sequential()
    num_layers = config['num_layers']
    lr = float(config['learning_rate'])
    l1_val = config.get('l1', 0.0) if config.get('use_l1', False) else 0.0
    l2_val = config.get('l2', 0.0) if config.get('use_l2', False) else 0.0
    regularizer = tf.keras.regularizers.l1_l2(l1=l1_val, l2=l2_val)
    
    for i in range(num_layers):
        units = config[f"units_{i+1}"]
        activation = config[f"activation_{i+1}"]
        if i == 0:
            model.add(Dense(units, kernel_regularizer=regularizer, input_shape=(input_dim,), use_bias=False))
        else:
            model.add(Dense(units, kernel_regularizer=regularizer, use_bias=False))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        if config["dropout"] > 0:
            model.add(Dropout(config["dropout"]))
    
    model.add(Dense(1, activation='sigmoid'))
    optimizers = {
        'adam': Adam(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'sgd': SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    }
    optimizer = optimizers.get(config['optimizer'], Adam(learning_rate=lr))
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.5, gamma=2.5),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

# --------------------------------------------
# 9. Definición del Espacio de Hiperparámetros (única función)
# --------------------------------------------
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 12

    # Hiperparámetros principales del clasificador
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.05, 0.5)
    batch_size = CategoricalHyperparameter('batch_size', [32, 64, 128, 256])
    optimizer = CategoricalHyperparameter('optimizer', ['adam', 'rmsprop', 'sgd'])
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size, optimizer])
    
    # Regularización para el clasificador
    l1_reg = UniformFloatHyperparameter('l1', 1e-6, 1e-2, log=True)
    l2_reg = UniformFloatHyperparameter('l2', 1e-6, 1e-2, log=True)
    use_l1 = CategoricalHyperparameter("use_l1", [True, False])
    use_l2 = CategoricalHyperparameter("use_l2", [True, False])
    cs.add_hyperparameters([l1_reg, l2_reg, use_l1, use_l2])
    cs.add_condition(EqualsCondition(l1_reg, use_l1, True))
    cs.add_condition(EqualsCondition(l2_reg, use_l2, True))
    
    # Hiperparámetros por capa para el clasificador
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])
        if i > 1:
            cs.add_condition(InCondition(child=units, parent=num_layers, values=list(range(i, max_layers+1))))
            cs.add_condition(InCondition(child=activation, parent=num_layers, values=list(range(i, max_layers+1))))
    
    # Hiperparámetros del Autoencoder Simple
    use_autoencoder = CategoricalHyperparameter("use_autoencoder", [True, False], default_value=True)
    ae_latent_dim = UniformIntegerHyperparameter("ae_latent_dim", 2, 50, default_value=10)
    ae_regularizacion = UniformFloatHyperparameter("ae_regularizacion", 1e-6, 1e-2, log=True, default_value=0.001)
    cs.add_hyperparameters([use_autoencoder, ae_latent_dim, ae_regularizacion])
    cs.add_condition(EqualsCondition(ae_latent_dim, use_autoencoder, True))
    cs.add_condition(EqualsCondition(ae_regularizacion, use_autoencoder, True))
    
    return cs

# --------------------------------------------
# 10. Función Objetivo para la Optimización de Hiperparámetros
# --------------------------------------------
global_history = []  # Almacena historiales de cada configuración

def objective_function(config, seed, budget):
    """
    Función objetivo para SMAC/BOHB.
    Se entrena (1) el autoencoder (si se activa) y (2) el clasificador usando
    StratifiedKFold. Se imprime la configuración para depuración.
    """
    print("\nConfiguración evaluada:", config)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    auc_pr_scores = []
    fold_histories = []
    
    global X_train_full, y_train_full

    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        X_train_fold = X_train_full.iloc[train_idx].values
        y_train_fold = y_train_full.iloc[train_idx].values
        X_val_fold = X_train_full.iloc[val_idx].values
        y_val_fold = y_train_full.iloc[val_idx].values
        
        # Preprocesamiento: Escalado y SMOTETomek
        X_train_scaled, y_train_bal, X_val_scaled, _ = preprocesar_datos(pd.DataFrame(X_train_fold), pd.Series(y_train_fold), pd.DataFrame(X_val_fold))
        
        # Si se activa el AE, entrenarlo y transformar los datos
        if config['use_autoencoder']:
            autoencoder, encoder = create_autoencoder_simple(config, input_dim=X_train_scaled.shape[1])
            autoencoder.fit(X_train_scaled, X_train_scaled, epochs=int(budget/2), batch_size=128, verbose=0)
            X_train_features = encoder.predict(X_train_scaled)
            X_val_features = encoder.predict(X_val_scaled)
        else:
            X_train_features = X_train_scaled
            X_val_features = X_val_scaled
        
        classifier = create_classifier(config, input_dim=X_train_features.shape[1])
        classifier.fit(X_train_features, y_train_bal, epochs=int(budget), batch_size=config["batch_size"], verbose=0)
        
        y_val_pred = classifier.predict(X_val_features, verbose=0).ravel()
        auc = average_precision_score(y_val_fold, y_val_pred)
        auc_pr_scores.append(auc)
        
        # Almacenar historial si es necesario
        history = classifier.history.history if hasattr(classifier, "history") else {}
        fold_histories.append(history)
    
    mean_auc = np.mean(auc_pr_scores)
    global_history.append({
        "config": config,
        "history": fold_histories,
        "mean_auc_pr": mean_auc
    })
    return 1 - mean_auc

# --------------------------------------------
# 11. Optimización de Hiperparámetros con BOHB/SMAC
# --------------------------------------------
def run_bohb_with_smac():
    """
    Ejecuta la optimización usando BOHB/SMAC.
    Se han aumentado los budgets: min_budget=20 y max_budget=100.
    """
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=300,
        min_budget=20,
        max_budget=100,
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
    Entrena el modelo final con la mejor configuración usando un 10% para validación.
    """
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    smote_tomek = SMOTETomek(sampling_strategy=0.8, random_state=42)
    X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train_split, y_train_split)
    
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_val_scaled = scaler.transform(X_val_split)
    
    model = create_classifier(best_config, input_dim=X_train_scaled.shape[1])
    
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
    
    y_val_pred = model.predict(X_val_scaled).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, best_threshold

# --------------------------------------------
# 13. Evaluación Final del Modelo
# --------------------------------------------
def evaluar_modelo(model, X_test, y_test, threshold):
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
# 14. Ejecución Principal
# --------------------------------------------
if __name__ == "__main__":
    print("Iniciando optimización de hiperparámetros...")
    best_config = run_bohb_with_smac()
    print("Optimización completada. Mejor configuración encontrada:")
    print(best_config)
    
    print("Entrenando el modelo final con la mejor configuración...")
    final_model, scaler, threshold = entrenar_modelo(best_config, X_train_full, y_train_full)
    
    X_test_scaled = scaler.transform(X_test)
    evaluar_modelo(final_model, X_test_scaled, y_test, threshold)
    
    # (Opcional) Imprimir el historial global de entrenamientos:
    for i, config_history in enumerate(global_history):
        print(f"\nConfiguración {i + 1}:")
        print(f"Hiperparámetros: {config_history['config']}")
        print(f"AUC-PR promedio: {config_history['mean_auc_pr']:.4f}")
