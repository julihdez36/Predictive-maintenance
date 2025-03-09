# 1. Importaciones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score, confusion_matrix, classification_report,
    PrecisionRecallDisplay, precision_recall_curve, f1_score, recall_score, precision_score,
    RocCurveDisplay, roc_auc_score, roc_curve, auc,
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
    # Transformación de los datos
    X_test_scaled = scaler.transform(X_test)
    X_test_reducido = pca.transform(X_test_scaled)
    
    # Predicciones
    y_pred_prob = model.predict(X_test_reducido).ravel()
    y_pred = (y_pred_prob >= best_threshold).astype(int)

    # Métricas
    auc_pr = average_precision_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Imprimir resultados
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Graficar la curva de Precisión-Recall
    plt.figure(figsize=(6, 6))
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob)
    plt.title(f"Curva de Precisión-Recall (AUC-PR = {auc_pr:.4f})")
    plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Devolver todas las variables importantes
    return {
        "Etiquetas reales": y_test,
        "Probabilidades": y_pred_prob,
        "Predicciones": y_pred
    }

    
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


if __name__ == "__main__":
    # Entrenar el modelo
    final_model, scaler, pca, threshold = entrenar_modelo_con_pca(best_config, X_train_full, y_train_full, variance_threshold=0.95)

    # Evaluar en el conjunto de prueba
    resultados_modelo = evaluar_modelo_con_pca(final_model, pca, scaler, X_test, y_test, threshold)

    # Extraer las variables necesarias para los gráficos
    y_true = resultados_modelo["Etiquetas reales"]
    y_pred_prob = resultados_modelo["Probabilidades"]
    y_pred = resultados_modelo["Predicciones"]



##########################################################
# Graficos
##########################################################

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

## 1. CURVA ROC
def graficar_roc(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()

## 2. CURVA PRECISION-RECALL
def graficar_pr(y_true, y_pred_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend(loc="upper right")
    plt.show()

## 3. MATRIZ DE CONFUSIÓN
def graficar_matriz_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.show()

## 4. HISTOGRAMA DE PROBABILIDADES CON EL UMBRAL
    
def graficar_histograma_probabilidades(y_true, y_pred_prob, threshold):
    plt.figure(figsize=(8, 6))

    # Filtrar las probabilidades por clase
    sns.histplot(y_pred_prob[y_true == 0], bins=20, kde=True, color="blue", label="Clase Negativa (0)", alpha=0.5)
    sns.histplot(y_pred_prob[y_true == 1], bins=20, kde=True, color="orange", label="Clase Positiva (1)", alpha=0.5)

    # Línea del umbral
    plt.axvline(threshold, color='red', linestyle='--', label=f'Umbral ({threshold:.2f})')

    # Configuración del gráfico
    plt.xlabel("Probabilidad Predicha")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Probabilidades por Clase")
    plt.legend()
    plt.show()

    
    

# Ejecutar funciones
graficar_roc(y_true, y_pred_prob)
graficar_pr(y_true, y_pred_prob)
graficar_matriz_confusion(y_true, y_pred)
graficar_histograma_probabilidades(y_true, y_pred_prob, threshold)


print(classification_report(y_true, y_pred))


