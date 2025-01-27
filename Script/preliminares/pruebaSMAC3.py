# Prueba SMAC (Sequential Model-based Algorithm Configuration).

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac import HyperparameterOptimizationFacade, Scenario
from smac.model.random_forest import RandomForest

# 1. Generar datos sintéticos
def generate_synthetic_data():
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 2. Definir el modelo de red neuronal
def create_model(learning_rate, num_units):
    model = Sequential()
    model.add(Dense(num_units, input_dim=6, activation='relu'))
    model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Función objetivo


def objective_function(config, seed):
    learning_rate = config['learning_rate']
    num_units = config['num_units']

    model = create_model(learning_rate, num_units)
    history = model.fit(X_train, y_train, epochs=3, batch_size=16, 
                        validation_data=(X_val, y_val), verbose=0)

    val_loss = history.history['val_loss'][-1]
    return val_loss


# 4. Espacio de búsqueda
def get_configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-3, 1e-2, log=True))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_units', 8, 32))
    return cs

# 5. Configurar y ejecutar SMAC3
def run_bohb_with_smac():
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=10,
        n_workers=4,  # Usar 4 workers para paralelizar
    )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=objective_function,
        model=RandomForest(configspace=get_configspace()),  # Pasar configspace aquí
    )

    incumbent = smac.optimize()
    return incumbent

# 6. Generar datos y ejecutar
X_train, X_val, y_train, y_val = generate_synthetic_data()
best_config = run_bohb_with_smac()


print("Mejores hiperparámetros encontrados:", best_config)








