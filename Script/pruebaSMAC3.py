# Prueba con SMAC3

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from smac.model.random_forest import RandomForest  # Importar RandomForest

# 1. Generar datos sintéticos pequeños
def generate_synthetic_data():
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 2. Definir el modelo de red neuronal
def create_model(learning_rate, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim=6, activation='relu'))  # 6 características de entrada
    model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=f'adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Definir la función objetivo
def objective_function(config):
    learning_rate = config['learning_rate']
    num_units = config['num_units']

    # Crear y entrenar el modelo
    model = create_model(learning_rate, num_units)
    history = model.fit(X_train, y_train, epochs=3, batch_size=16, validation_data=(X_val, y_val), verbose=0)

    # Devolver la pérdida en validación
    val_loss = history.history['val_loss'][-1]
    return val_loss

# 4. Definir el espacio de búsqueda reducido
def get_configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-3, 1e-2, log=True))  # Rango pequeño
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_units', 8, 32))  # Rango pequeño
    return cs

# 5. Configurar y ejecutar BOHB con SMAC3
def run_bohb_with_smac():
    # Definir el escenario de optimización
    scenario = Scenario(
        configspace=get_configspace(),  # Espacio de búsqueda
        deterministic=True,             # Asume que la función objetivo es determinística
        n_trials=10,                    # Número de evaluaciones de hiperparámetros
        n_workers=1,                    # Número de workers en paralelo
    )

    # Crear el objeto de optimización (usando BOHB)
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_function,
        model=RandomForest(),  # Usar bosques de regresión aleatoria
    )

    # Ejecutar la optimización
    incumbent = smac.optimize()

    return incumbent

# 6. Generar datos sintéticos
X_train, X_val, y_train, y_val = generate_synthetic_data()

# 7. Ejecutar BOHB con SMAC3
best_config = run_bohb_with_smac()
print("Mejores hiperparámetros encontrados:", best_config)