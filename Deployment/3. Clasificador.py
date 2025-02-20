# ============================================
# Parte 2: Entrenamiento y Evaluación del Clasificador
# ============================================
"""
Este script carga la mejor configuración (almacenada en 'hiperparametros.json'),
entrena el modelo final utilizando PCA (conservando el 95% de la varianza) y evalúa el clasificador.
"""

# 1. Importaciones
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys

from sklearn.model_selection import train_test_split
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

from imblearn.combine import SMOTETomek

# 2. Configuración Inicial y Datos
tf.random.set_seed(42)
np.random.seed(42)

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_final = pd.read_csv(url)
X = df_final.drop(columns=['failed'])
y = df_final['failed']

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Funciones de Preprocesamiento y PCA
def preprocesar_datos_con_pca(X_train_scaled, y_train, pca, aplicar_smote=False):
    X_train_reducido = pca.transform(X_train_scaled)
    if aplicar_smote:
        smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42)
        X_train_reducido, y_train_bal = smote_tomek.fit_resample(X_train_reducido, y_train)
    else:
        y_train_bal = y_train
    return X_train_reducido, y_train_bal

def entrenar_pca(X_train_scaled, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold, random_state=42)
    pca.fit(X_train_scaled)
    return pca

# 4. Callback OneCycleLR (igual que en Parte 1)
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

# 6. Arquitectura del Modelo
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
        'sgd': SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    }
    optimizer = optimizers.get(best_config['optimizer'].lower(), Adam(learning_rate=lr))
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha= 0.95, gamma= 5),
        metrics=[tf.keras.metrics.AUC(name='auc_pr', curve='PR')]
    )
    return model

def find_best_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    return thresholds[np.nanargmax(f1_scores)]

# 7. Función para Entrenar el Modelo Final con la Mejor Configuración
def entrenar_modelo_con_pca(best_config, X_train_full, y_train_full, variance_threshold=0.95):
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train_full, test_size=0.1, stratify=y_train_full, random_state=42
    )
    
    smote_tomek = SMOTETomek(sampling_strategy=1, random_state=42) ###################################
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
    
    class_weights = {0: 1, 1: 10}
    model.fit(
        X_train_reducido, y_train_final,
        validation_data=(X_val_reducido, y_val_split),
        epochs=100,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    y_val_pred = model.predict(X_val_reducido).ravel()
    best_threshold = find_best_threshold(y_val_split, y_val_pred)
    
    return model, scaler, pca, best_threshold

# 8. Función para Evaluar el Modelo Final
def evaluar_modelo_con_pca(model, pca, scaler, X_test, y_test, best_threshold):
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
    
    return y_pred

# 9. (Opcional) Función para Graficar la Matriz de Confusión
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
    
best_config = {
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

# 10. Ejecución Principal
if __name__ == "__main__":
    # Entrenar el modelo final
    final_model, scaler, pca, threshold = entrenar_modelo_con_pca(best_config, X_train_full, y_train_full, variance_threshold=0.95)
    print("Umbral óptimo obtenido:", threshold)
    
    # Evaluar en el conjunto de prueba
    evaluar_modelo_con_pca(final_model, pca, scaler, X_test, y_test, threshold)
    
    # (Opcional) Graficar la matriz de confusión
    y_pred = final_model.predict(pca.transform(scaler.transform(X_test))).ravel()
    y_pred_bin = (y_pred >= threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred_bin, classes=["No failed", "Failed"])
    
    # (Opcional) Guardar artefactos del experimento
    resultados_evaluacion = {
        "auc_pr": average_precision_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred_bin),
        "recall": recall_score(y_test, y_pred_bin),
        "f1": f1_score(y_test, y_pred_bin)
    }
    with open("resultados_evaluacion.json", "w") as f:
        json.dump(resultados_evaluacion, f, indent=4)
    print("Resultados de evaluación guardados en 'resultados_evaluacion.json'.")
    
    final_model.save("Modelos/final_model.h5")
    print("El modelo final se ha guardado en 'final_model.h5'.")
    
    metadatos_experimento = {
        "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "semilla": 42,
        "versiones": {
            "tensorflow": tf.__version__,
            "python": sys.version
        }
    }
    with open("Modelos/metadatos_experimento.json", "w") as f:
        json.dump(metadatos_experimento, f, indent=4)
    print("Metadatos del experimento guardados en 'metadatos_experimento.json'.")
