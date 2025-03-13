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


round(df_final.failed.value_counts(normalize = False), 4)

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

import os


# Crear una carpeta para guardar los gráficos si no existe
output_dir = "Graficos"
os.makedirs(output_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})


## 1. ROC Curve
def plot_roc_curve(y_true, y_pred_prob, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

## 2. Precision-Recall Curve
def plot_precision_recall_curve(y_true, y_pred_prob, filename="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = average_precision_score(y_true, y_pred_prob)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

## 3. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"], 
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

## 4. Histogram of Predicted Probabilities with Threshold
def plot_probability_histogram(y_true, y_pred_prob, threshold, filename="probability_histogram.png"):
    plt.figure(figsize=(8, 6))

    # Plot distributions for each class
    sns.histplot(y_pred_prob[y_true == 0], bins=20, kde=True, color="blue", label="Negative Class (0)", alpha=0.5)
    sns.histplot(y_pred_prob[y_true == 1], bins=20, kde=True, color="orange", label="Positive Class (1)", alpha=0.5)

    # Threshold line
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')

    # Labels and title
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    # plt.title("Probability Distribution by Class")
    plt.legend()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

    
    

# Ejecutar funciones
plot_roc_curve(y_true, y_pred_prob)
plot_precision_recall_curve(y_true, y_pred_prob)
plot_confusion_matrix(y_true, y_pred)
plot_probability_histogram(y_true, y_pred_prob, threshold)


print(classification_report(y_true, y_pred))

def plot_roc_pr_curves(y_true, y_pred_prob, filename="roc_pr_curves.png"):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auc_pr = average_precision_score(y_true, y_pred_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 fila, 2 columnas

    # ROC Curve
    axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.3f}')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='green', lw=2, label=f"AUC-PR = {auc_pr:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.tight_layout()  # Ajustar diseño
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

plot_roc_pr_curves(y_true, y_pred_prob)


#### Resultados aceptando un 10 y un 1% de falsos descubrimientos

def ajustar_umbral_con_fdr(y_true, y_pred_prob, fdr_threshold=0.01):
    # Calcular la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

    # Calcular FDR (False Discovery Rate)
    # FDR = FP / (FP + TP)
    false_positives = (1 - precision) * np.sum(y_true == 0)  # Falsos positivos
    true_positives = precision * np.sum(y_true == 1)         # Verdaderos positivos
    fdr = false_positives / (false_positives + true_positives)

    # Encontrar el umbral que minimiza FDR para el umbral deseado
    # Buscamos el primer umbral donde la FDR es menor que el umbral tolerado (1% o 10%)
    threshold_ajustado = thresholds[np.where(fdr <= fdr_threshold)[0][0]]

    print(f"Umbral ajustado para FDR < {fdr_threshold*100}%: {threshold_ajustado:.3f}")
    return threshold_ajustado

# Ejemplo de uso
threshold_1 = ajustar_umbral_con_fdr(y_true, y_pred_prob, fdr_threshold=0.01)
threshold_10 = ajustar_umbral_con_fdr(y_true, y_pred_prob, fdr_threshold=0.10)


def evaluate_at_fdr_threshold(y_true, y_pred_prob, threshold, fdr_target):
    y_pred = (y_pred_prob >= threshold).astype(int)
    recall = recall_score(y_true, y_pred)
    print(f"At FDR {fdr_target*100:.0f}%, the model detects {recall*100:.2f}% of failures.")
    return recall

recall_fdr_1 = evaluate_at_fdr_threshold(y_true, y_pred_prob, threshold_1, 0.01)
recall_fdr_10 = evaluate_at_fdr_threshold(y_true, y_pred_prob, threshold_10, 0.10)

