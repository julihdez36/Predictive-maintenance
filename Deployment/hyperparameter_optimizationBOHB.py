

####################################################################

# división y balanceo de datos

# SMOTE (sobremuestreo de la clase minoritaria)
# Tomek Links (submuestreo de la clase mayoritaria)


import pandas as pd
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.model_selection import train_test_split

# Cargar datos
df_final = pd.read_csv('Data/df_entrenamiento.csv')
X = df_final.drop(columns=['burned_transformers'])  
y = df_final['burned_transformers']

# 1. Dividir en entrenamiento (80%) y prueba (20%) - estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Mantener proporción original en train y test
)

# 2. Dividir el entrenamiento en subtrain (80%) y validación (20%) - estratificado
X_subtrain, X_val, y_subtrain, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train  # Mantener proporción original en subtrain y val
)

# 3. Aplicar SMOTE + Tomek solo al subtrain (no a validación o prueba)
smote_tomek = SMOTETomek(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_subtrain, y_subtrain)

# Verificar distribuciones
print("Distribución original (y):", Counter(y))
print("Distribución subtrain antes de SMOTE:", Counter(y_subtrain))
print("Distribución subtrain después de SMOTE:", Counter(y_resampled))
print("Distribución validación:", Counter(y_val))  # Debe mantener proporción original
print("Distribución prueba:", Counter(y_test))     # Debe mantener proporción original


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
y_url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/y_resampled.csv'

X_train = pd.read_csv(X_url)
y_train = pd.read_csv(y_url)

# Conversión en array y normalización de X

y_train = y_train.values.ravel()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# Dividir el conjunto balanceado en entrenamiento y validación
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, 
    y_train, 
    test_size=0.2,  # 20% para validación
    random_state=42,
    stratify=y_train  # Mantener proporción de clases
)


# 2. Definición del modelo
def create_model(config):
    model = Sequential()
    num_layers = config["num_layers"]
    
    # Primera capa
    model.add(Dense(
        units=config["units_1"],
        input_dim=X_train.shape[1],
        activation=config["activation_1"]
    ))
    model.add(Dropout(config["dropout"]))

    # Capas ocultas (si num_layers > 1)
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

# 3. Función objetivo para SMAC
def objective_function(config, instance, seed, budget):
    model = create_model(config)
    history = model.fit(
        X_train, y_train,
        epochs=int(budget),
        batch_size=config["batch_size"],
        validation_data=(X_val, y_val),
        verbose=0
    )
    return history.history['val_loss'][-1]

# 4. Espacio de búsqueda de hiperparámetros
def get_configspace():
    cs = ConfigurationSpace()
    max_layers = 6

    # Hiperparámetros base
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 1e-2, log=True)
    num_layers = UniformIntegerHyperparameter('num_layers', 1, max_layers)
    dropout = UniformFloatHyperparameter('dropout', 0.1, 0.5)
    batch_size = UniformIntegerHyperparameter('batch_size', 32, 256, log=True)
    cs.add_hyperparameters([learning_rate, num_layers, dropout, batch_size])

    # Hiperparámetros por capa
    for i in range(1, max_layers + 1):
        units = UniformIntegerHyperparameter(f"units_{i}", 20, 100, log=True)
        activation = CategoricalHyperparameter(f"activation_{i}", ['relu', 'tanh', 'swish'])
        cs.add_hyperparameters([units, activation])

        # Condiciones para capas superiores
        if i > 1:
            cs.add_condition(GreaterThanCondition(units, num_layers, i - 1))
            cs.add_condition(GreaterThanCondition(activation, num_layers, i - 1))

    return cs

# 5. Configuración y ejecución de SMAC
def run_bohb_with_smac():
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

# 6. Ejecutar la optimización
if __name__ == "__main__":
    best_config = run_bohb_with_smac()
    print("\nMejor configuración encontrada:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
