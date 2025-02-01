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
import json
import joblib
from keras.utils import plot_model

# Semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# Carga de datos
df_final = pd.read_csv('Data/df_entrenamiento.csv')

X = df_final.drop(columns=['burned_transformers'])  
y = df_final['burned_transformers']

# Split inicial en train (80%) y test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocesamiento: balanceo y escalada de datos
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

# Para la optimización: preproceso sobre entrenamiento

X_train_bal, y_train_bal, scaler_cv = preprocesar_datos(X_train_full, y_train_full)

# Escalamiento del conjunto de prueba 
X_test_scaled = scaler_cv.transform(X_test)

# Función del modelo (dinámico) de red

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

# 4. Función objetivo: SMAC con validación cruzada y early stopping

def objective_function(config, seed, budget):  # Eliminamos el argumento 'instance'
    # Usamos 3-fold cross validation sobre el conjunto de entrenamiento balanceado
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv_losses = []
    
    for train_idx, val_idx in kfold.split(X_train_bal, y_train_bal):
        X_train_cv, X_val_cv = X_train_bal[train_idx], X_train_bal[val_idx]
        y_train_cv, y_val_cv = np.array(y_train_bal)[train_idx], np.array(y_train_bal)[val_idx]
        
        model = create_model(config)
        
        # Callback de early stopping: monitorizar la pérdida en validación 5 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train_cv, y_train_cv,
            epochs=int(budget),
            batch_size=config["batch_size"],
            validation_data=(X_val_cv, y_val_cv),
            verbose=0,
            callbacks=[early_stop]
        )
        
        # Pérdida de validación del último epoch (o del mejor restaurado)
        cv_losses.append(history.history['val_loss'][-1])
    
    # promedio de pérdidas de validación de los folds
    return np.mean(cv_losses)

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
        n_trials=300,      # n. evaluaciones
        min_budget=5,      # n. mínimo de epochs en cada evaluación
        max_budget=50,     # n. máximo de epochs en cada evaluación
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

# modelo final con la mejor configuración encontrada
final_model = create_model(best_config)
# Callback de early stopping para el entrenamiento final
early_stop_final = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1)

# Entrenar el modelo final 
final_model.fit(
    X_train_final, y_train_final,
    epochs=50,
    batch_size=best_config["batch_size"],
    verbose=1,
    callbacks=[early_stop_final]
)

# Evaluación del modelo final
loss, accuracy = final_model.evaluate(X_test_final, y_test, verbose=0)
print(f"\nPérdida en test: {loss:.4f}")
print(f"Exactitud en test: {accuracy:.4f}")

# Guardado de modelo

final_model.save('modelo_final.h5')


best_config_dict = best_config.get_dictionary()

with open('mejores_hiperparametros.json', 'w') as f:
    json.dump(best_config_dict, f, indent=4)
joblib.dump(scaler_final, 'scaler_entrenado.pkl')

# Gráfico
# plot_model(
#     final_model,
#     to_file='model_plot.png',
#     show_shapes=True,
#     show_layer_names=True,
#     rankdir='TB',
#     dpi=96
# )