# Librerías de trabajo

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import plot_model
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import GreaterThanCondition
from smac.facade import HyperbandFacade
from smac import Scenario
import tensorflow as tf
from imblearn.combine import SMOTETomek
import json
import joblib
from skopt import gp_minimize

# Semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Paralelicemos la ejecución

tf.config.threading.set_inter_op_parallelism_threads(4)  # Ajusta según tus núcleos
tf.config.threading.set_intra_op_parallelism_threads(4)

print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())



# Carga de datos
url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento.csv'
df_final = pd.read_csv(url)

X = df_final.drop(columns=['burned_transformers'])  
y = df_final['burned_transformers']

# Split inicial en train (80%) y test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocesamiento: balanceo escalada de datos

def preprocesar_datos(X, y, scaler_fit=True, scaler_obj=None):
    # Aplicar SMOTE + Tomek para balancear
    smote_tomek = SMOTETomek(sampling_strategy=0.5, random_state=42)
    X_res, y_res = smote_tomek.fit_resample(X, y)
    
    # Escalar
    if scaler_fit or (scaler_obj is None):
        scaler = StandardScaler()
        X_res = scaler.fit_transform(X_res)
    else:
        scaler = scaler_obj
        X_res = scaler.transform(X_res)
    return X_res, y_res, scaler

# Conjutno de entrenamiento balanceado
# X_(34978,21)

X_train_bal, y_train_bal, scaler_cv = preprocesar_datos(X_train_full, y_train_full)

# Escalamiento del conjunto de prueba 
X_test_scaled = scaler_cv.transform(X_test)


# Red neuronal

def create_model(config):
    model = Sequential()
    num_layers = config["num_layers"]
    
    # Primera capa
    model.add(Dense(
        units=config["units_1"],
        input_dim=X_train_bal.shape[1],
        activation=config["activation_1"]
    ))
    model.add(Dropout(config["dropout"]))

    # Capas ocultas adicionales
    for i in range(2, num_layers + 1):
        model.add(Dense(
            units=config[f"units_{i}"],
            activation=config[f"activation_{i}"]
        ))
        model.add(Dropout(config["dropout"]))

    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Función objetivo para la optimización de hiperparámetros

def objective_function(config, seed, budget):
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_aucs_pr = []  # Guardar los AUC-PR de cada fold

    for train_idx, val_idx in kfold.split(X_train_bal, y_train_bal):
        X_train_cv, X_val_cv = X_train_bal[train_idx], X_train_bal[val_idx]
        y_train_cv, y_val_cv = np.array(y_train_bal)[train_idx], np.array(y_train_bal)[val_idx]

        model = create_model(config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name="auc_pr", curve="PR")]  # <== Cambia de ROC a PR
        )

        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_auc_pr', patience=5, restore_best_weights=True, mode='max', verbose=0)

        history = model.fit(
            X_train_cv, y_train_cv,
            epochs=int(budget),
            batch_size=config["batch_size"],
            validation_data=(X_val_cv, y_val_cv),
            verbose=0,
            callbacks=[early_stop]
        )

        best_auc_pr = max(history.history['val_auc_pr'])  # Extraer el mejor AUC-PR
        cv_aucs_pr.append(best_auc_pr)

    return -np.mean(cv_aucs_pr)  # Negativo porque optimizamos minimizando



# Espacio de búsqueda de hiperparámetros

def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 7

    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.1, 0.5)
    batch_size = UniformIntegerHyperparameter('batch_size', 32, 256, log=True)
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size])

    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 250, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish', 'elu'])
        cs.add_hyperparameters([units, activation])

        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))

    return cs

# 6. ejecución de BOHB con SMAC

def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=300,      # n. evaluaciones (300)
        min_budget=5,      # n. mínimo de epochs en cada evaluación
        max_budget=50,     # n. máximo de epochs en cada evaluación (50)
        n_workers=4
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )

    return smac.optimize()


# Ejecución BOBH

best_config = run_bohb_with_smac()
print("\nMejor configuración encontrada:")
for key, value in best_config.items():
    print(f"{key}: {value}")

# Modelo
# Conjunto de entrenamiento completo
X_train_final, y_train_final, scaler_final = preprocesar_datos(X_train_full, y_train_full, scaler_fit=True)
# conjunto de prueba con el scaler final
X_test_final = scaler_final.transform(X_test)

# Entrenamiento del modelo final

final_model = create_model(best_config)
early_stop_final = tf.keras.callbacks.EarlyStopping(monitor='auc_pr', patience=5, restore_best_weights=True, verbose=1)

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=best_config["learning_rate"]),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name="auc_pr", curve="PR")]
)

final_model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=best_config["batch_size"],
    verbose=1,
    callbacks=[early_stop_final]
)



# ---- 1 Evaluación final del modelo ----
loss, accuracy = final_model.evaluate(X_test_final, y_test, verbose=0)
print(f"\n Pérdida en test: {loss:.4f}")
print(f" Exactitud en test: {accuracy:.4f}")

# Predicciones de probabilidades
y_pred_prob = final_model.predict(X_test_final)

# ---- 2️Encontrar umbral óptimo basado en F1-score ----
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, (y_pred_prob > t).astype(int)) for t in thresholds]
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"\n Umbral óptimo basado en F1-score: {optimal_threshold:.2f}")

# Aplicar umbral óptimo a las predicciones
y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)

# ---- 3 Calcular métricas avanzadas ----
# Curva Precision-Recall y AUC-PR
precision_pr, recall_pr, _ = precision_recall_curve(y_test, y_pred_prob)
auc_pr = average_precision_score(y_test, y_pred_prob)
print(f"AUC-PR (Precision-Recall): {auc_pr:.4f}")

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_roc = auc(fpr, tpr)
print(f"🔹 AUC-ROC: {auc_roc:.4f}")

# ---- 4️⃣ Visualización de resultados ----

# 📌 Matriz de confusión con etiquetas
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Quemado', 'Quemado'], 
            yticklabels=['No Quemado', 'Quemado'])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# 📌 Gráfica de F1-score vs. umbral
plt.figure(figsize=(6, 4))
plt.plot(thresholds, f1_scores, marker='o', linestyle='-')
plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Óptimo: {optimal_threshold:.2f}')
plt.xlabel('Umbral de decisión')
plt.ylabel('F1-score')
plt.title('F1-score vs. Umbral')
plt.legend()
plt.grid()
plt.show()

# 📌 Curva Precision-Recall
plt.figure(figsize=(6, 4))
plt.plot(recall_pr, precision_pr, marker='.', label=f'AUC-PR: {auc_pr:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid()
plt.show()

# 📌 Curva ROC
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, marker='.', label=f'AUC-ROC: {auc_roc:.2f}')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend()
plt.grid()
plt.show()


# Guardado de modelo

final_model.save('Modelos/segundo_modeloAUC.h5')

best_config_dict = best_config.get_dictionary()

with open('Modelos/segundo_hiperparametrosAUC.json', 'w') as f:
    json.dump(best_config_dict, f, indent=4)
    
joblib.dump(scaler_final, 'Modelos/segundo_scalerAUC.pkl')

