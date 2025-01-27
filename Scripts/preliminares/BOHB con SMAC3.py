## Código de BOHB con SMAC3 (Hyperband)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario

# 1. Generar datos sintéticos
def generate_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 2. Definir el modelo
def create_model(learning_rate, num_units, num_layers, activation):
    model = Sequential()
    model.add(Dense(num_units, input_dim=20, activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Función objetivo

def objective_function(config, budget, seed):
    learning_rate = config["learning_rate"]
    num_units = config["num_units"]
    num_layers = config["num_layers"]
    activation = config["activation"]

    model = create_model(learning_rate, num_units, num_layers, activation)
    history = model.fit(X_train, y_train, epochs=int(budget), batch_size=32, validation_data=(X_val, y_val), verbose=0)

    val_loss = history.history['val_loss'][-1]
    return float(val_loss)

# 4. Definir el espacio de búsqueda
def get_configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-4, 1e-2, log=True))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_units', 16, 128))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers', 1, 4))
    cs.add_hyperparameter(CategoricalHyperparameter('activation', ['relu', 'tanh']))
    return cs

# 5. Configurar y ejecutar BOHB con SMAC3
def run_bohb_with_smac():
    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = generate_synthetic_data()

    # Configurar el escenario
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=50,  # Número total de evaluaciones
        n_workers=4,  # Número de workers en paralelo
        min_budget=1,  # Mínimo número de épocas
        max_budget=20,  # Máximo número de épocas
    )

    # Ejecutar BOHB
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_function,
    )
    incumbent = smac.optimize()

    return incumbent

# 6. Ejecutar BOHB
if __name__ == "__main__":
    best_config = run_bohb_with_smac()
    print("Mejores hiperparámetros encontrados:", best_config)