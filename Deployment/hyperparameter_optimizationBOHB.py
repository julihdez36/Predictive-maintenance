import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import GreaterThanCondition
from smac.facade import HyperbandFacade
from smac import Scenario
import tensorflow as tf
from imblearn.combine import SMOTETomek

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# 1. Cargar datos y dividir en entrenamiento, validación y prueba
df_final = pd.read_csv('Data/df_entrenamiento.csv')

X = df_final.drop(columns=['burned_transformers'])  
y = df_final['burned_transformers']

# Split inicial en train (80%) y test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# En este caso, para la optimización usaremos solo el conjunto de entrenamiento (X_train_full, y_train_full)

# 2. Función para balancear y escalar datos
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

# Para la optimización: balanceamos y escalamos el conjunto de entrenamiento
X_train_bal, y_train_bal, scaler_cv = preprocesar_datos(X_train_full, y_train_full)

# Escalar el conjunto de prueba con el mismo scaler (aunque en la optimización no se usa test)
X_test_scaled = scaler_cv.transform(X_test)

# 3. Definición del modelo
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

# 4. Función objetivo para SMAC con validación cruzada y early stopping

def objective_function(config, instance, seed, budget):
    # Usamos 5-fold cross validation sobre el conjunto de entrenamiento balanceado
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_losses = []
    
    for train_idx, val_idx in kfold.split(X_train_bal, y_train_bal):
        X_train_cv, X_val_cv = X_train_bal[train_idx], X_train_bal[val_idx]
        y_train_cv, y_val_cv = np.array(y_train_bal)[train_idx], np.array(y_train_bal)[val_idx]
        
        model = create_model(config)
        
        # Callback de early stopping: se monitoriza la pérdida en validación con paciencia de 5 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train_cv, y_train_cv,
            epochs=int(budget),
            batch_size=config["batch_size"],
            validation_data=(X_val_cv, y_val_cv),
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Se toma la pérdida de validación del último epoch (o del mejor restaurado)
        cv_losses.append(history.history['val_loss'][-1])
    
    # Retornamos el promedio de las pérdidas de validación de los folds
    return np.mean(cv_losses)

# 5. Definición del espacio de búsqueda de hiperparámetros
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 7

    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.1, 0.5)
    batch_size = UniformIntegerHyperparameter('batch_size', 15, 200, log=True)
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size])

    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 10, 150, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish'])
        cs.add_hyperparameters([units, activation])

        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))

    return cs

# 6. Ejecutar BOHB (BOHB a través de SMAC)
def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=150,      # Número de evaluaciones
        min_budget=5,      # Número mínimo de epochs en cada evaluación
        max_budget=50,     # Número máximo de epochs en cada evaluación
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )

    return smac.optimize()

# 7. Entrenamiento final del modelo con la mejor configuración
if __name__ == "__main__":
    # Ejecutar BOHB para encontrar la mejor configuración
    best_config = run_bohb_with_smac()
    print("\nMejor configuración encontrada:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    
    # Para el entrenamiento final, combinamos todos los datos de entrenamiento (antes se tenía solo X_train_full)
    # Re-balanceamos y re-escalamos usando TODO el conjunto de entrenamiento (excluyendo el test)
    X_train_final, y_train_final, scaler_final = preprocesar_datos(X_train_full, y_train_full, scaler_fit=True)
    # Escalar el conjunto de prueba con el scaler final
    X_test_final = scaler_final.transform(X_test)
    
    # Crear el modelo final con la mejor configuración encontrada
    final_model = create_model(best_config)
    # Callback de early stopping para el entrenamiento final
    early_stop_final = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)
    
    # Entrenar el modelo final (se puede ajustar el número de epochs; aquí usamos 50 como ejemplo)
    final_model.fit(
        X_train_final, y_train_final,
        epochs=50,
        batch_size=best_config["batch_size"],
        verbose=1,
        callbacks=[early_stop_final]
    )
    
    # Evaluar el modelo final en el conjunto de prueba (datos reales, no balanceados)
    loss, accuracy = final_model.evaluate(X_test_final, y_test, verbose=0)
    print(f"\nPérdida en test: {loss:.4f}")
    print(f"Exactitud en test: {accuracy:.4f}")
    
    
# Métricas adicionales
    
from sklearn.metrics import classification_report

y_pred = final_model.predict(X_test_final)
y_pred_classes = (y_pred > 0.5).astype(int)  # Convertir probabilidades a clases binarias
print(classification_report(y_test, y_pred_classes))


# Guardemos el modelo

# Guardar el modelo entrenado
final_model.save('Modelos/modelo1_entrenado.h5')

# guardemos también los hiperparámetros

import json

# Convertir a diccionario
best_config_dict = best_config.get_dictionary()

# Guardar en JSON
with open('Modelos/hiper_modelo1.json', 'w') as f:
    json.dump(best_config_dict, f, indent=4)

    
# Guardar el scaler
import joblib

joblib.dump(scaler_final, 'Modelos/scaler_m1.pkl')


# Para cargarlo todo

# from keras.models import load_model
# import json
# import joblib

# Cargar el modelo entrenado
# modelo_cargado = load_model('mi_modelo_entrenado.h5')

# Cargar los hiperparámetros
# with open('mejores_hiperparametros.json', 'r') as f:
    # best_config_cargado = json.load(f)

# Cargar el scaler
# scaler_cargado = joblib.load('scaler_entrenado.pkl')



## Visualización del modelo

from keras.utils import plot_model

# Graficar el modelo
plot_model(
    final_model,  # Tu modelo de Keras
    to_file='model_plot.png',  # Archivo de salida
    show_shapes=True,  # Mostrar formas de los tensores
    show_layer_names=True,  # Mostrar nombres de las capas
    rankdir='TB',  # Orientación del gráfico: 'TB' (vertical), 'LR' (horizontal)
    dpi=96  # Resolución de la imagen
)