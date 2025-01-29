
## Definición de hiperparámetros

# 0. Librerías

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import GreaterThanCondition
from smac.facade import HyperbandFacade
from smac import Scenario
import tensorflow as tf

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)



# 1. Carga de datos y ajuste

X_url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/X_resampled.csv'
y_url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/y_resampled.shape.csv'

X = pd.read_csv(X_url)
y = pd.read_csv(y_url)


def load_data():
    global X_train, X_val, y_train, y_val
    
    # Cargar datos desde los CSVs
    X = pd.read_csv(X_url)
    y = pd.read_csv(y_url)

    # Si `y` es un DataFrame con una sola columna, convertir a array
    y = y.values.ravel()

    # Normalizar X para mejorar la convergencia del modelo
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dividir en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Definir la creación del modelo
def create_model(config):
    model = Sequential()
    num_layers = config["num_layers"]
    
    # Primera capa
    model.add(Dense(
        units=config["units_1"],
        input_dim=X_train.shape[1],  # Asegurar que use la dimensión correcta
        activation=config["activation_1"]
    ))
    model.add(Dropout(config["dropout"]))

    # Capas ocultas
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

# 3. Función objetivo para optimización
def objective_function(config, instance, seed, budget):
    global X_train, X_val, y_train, y_val

    model = create_model(config)
    history = model.fit(
        X_train, y_train,
        epochs=int(budget),
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history.history['val_loss'][-1]  # Optimizar con la pérdida de validación

# 4. Espacio de búsqueda optimizado
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 6

    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.1, 0.5)
    batch_size = UniformIntegerHyperparameter('batch_size', 32, 256, log=True)
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size])

    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 20, 100, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish'])
        cs.add_hyperparameters([units, activation])

        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))

    return cs

# 5. Ejecutar optimización con SMAC
def run_bohb_with_smac():
    load_data()

    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=150,
        min_budget=5,
        max_budget=50,
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )

    return smac.optimize()

# 6. Ejecutar optimización
if __name__ == "__main__":
    best_config = run_bohb_with_smac()
    print("\nMejor configuración encontrada:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
