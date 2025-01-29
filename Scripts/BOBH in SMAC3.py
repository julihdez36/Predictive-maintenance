import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from smac.facade import HyperbandFacade  # Corregido: Usar HyperbandFacade
from smac import Scenario
import tensorflow as tf  # Para control de semillas

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

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

# 3. Función objetivo (corregir firma)
def objective_function(config, instance, seed, budget):  # Agregar 'instance'
    # Extraer hiperparámetros
    learning_rate = config["learning_rate"]
    num_units = config["num_units"]
    num_layers = config["num_layers"]
    activation = config["activation"]

    # Crear y entrenar el modelo con el "budget"
    model = create_model(learning_rate, num_units, num_layers, activation)
    history = model.fit(
        X_train, y_train,
        epochs=int(budget),
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # Obtener la pérdida de validación
    val_loss = history.history['val_loss'][-1]
    return val_loss

# 4. Espacio de búsqueda
def get_configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-4, 1e-2, log=True))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_units', 16, 128))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_layers', 1, 4))
    cs.add_hyperparameter(CategoricalHyperparameter('activation', ['relu', 'tanh']))
    return cs

# 5. Configurar y ejecutar BOHB
def run_bohb_with_smac():
    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = generate_synthetic_data()

    # Configurar el escenario
    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=50,
        min_budget=1,
        max_budget=20,
    )

    # Usar HyperbandFacade
    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True,
    )

    incumbent = smac.optimize()
    return incumbent

# 6. Ejecutar
if __name__ == "__main__":
    best_config = run_bohb_with_smac()
    print("Mejores hiperparámetros encontrados:", best_config)
    

######################################################## 
########################################################   
# Personalización de capas

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import GreaterThanCondition
from smac.facade import HyperbandFacade
from smac import Scenario
import tensorflow as tf

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

# 1. Generar datos sintéticos
def generate_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 2. Nuevo modelo con hiperparámetros por capa
def create_model(config):
    model = Sequential()
    num_layers = config["num_layers"]
    
    # Primera capa (con input_dim)
    model.add(Dense(
        units=config["units_1"],
        input_dim=20,
        activation=config["activation_1"]
    ))
    
    # Capas subsiguientes
    for i in range(2, num_layers + 1):
        model.add(Dense(
            units=config[f"units_{i}"],
            activation=config[f"activation_{i}"]
        ))
    
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = Adam(learning_rate=config["learning_rate"])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Función objetivo modificada
def objective_function(config, instance, seed, budget):
    global X_train, X_val, y_train, y_val  # Asegurar que las variables son accesibles

    model = create_model(config)
    history = model.fit(
        X_train, y_train,
        epochs=int(budget),
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history.history['val_loss'][-1]

# 4. Nuevo espacio de búsqueda jerárquico
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 4  # Máximo número de capas a considerar
    
    # Hiperparámetros globales
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-4, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    cs.add_hyperparameters([learning_rate, num_layers])
    
    # Hiperparámetros por capa
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 16, 128)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh'])
        
        cs.add_hyperparameters([units, activation])
        
        # Condiciones para capas superiores a 1
        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))
    
    return cs

# 5. Configurar y ejecutar BOHB
def run_bohb_with_smac():
    global X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = generate_synthetic_data()

    scenario = Scenario(
        configspace=get_configspace(),
        deterministic=True,
        n_trials=100,  # Aumentar pruebas para espacio más complejo
        min_budget=1,
        max_budget=20,
    )

    smac = HyperbandFacade(
        scenario=scenario,
        target_function=objective_function,
        overwrite=True
    )

    return smac.optimize()

# 6. Ejecutar
if __name__ == "__main__":
    best_config = run_bohb_with_smac()
    print("\nMejor configuración encontrada:")
    for key, value in best_config.items():
        print(f"{key}: {value}")

