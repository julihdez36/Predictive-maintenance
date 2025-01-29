# Implementación BOHB con hpbandster

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter

# 1. Función para la generación de datos sintéticos

def generate_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# 2. Definimos el modelo de red neuronal

def create_model(learning_rate, num_units, dropout_rate):
    model = Sequential()
    model.add(Dense(num_units, input_dim=20, activation='relu'))
    model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=f'adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Definir la función objetivo para BOHB

def objective_function(hyperparameters):
    learning_rate = hyperparameters['learning_rate']
    num_units = hyperparameters['num_units']
    dropout_rate = hyperparameters['dropout_rate']

    # Crear y entrenar el modelo
    model = create_model(learning_rate, num_units, dropout_rate)
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Devolver la pérdida en validación
    val_loss = history.history['val_loss'][-1]
    return {'loss': val_loss}

# 4. Definir un Worker personalizado

class MyWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, **kwargs):
        result = objective_function(config)
        return result

# 5. Definición del espacio de búsqueda de hiperparámetros
def get_configspace():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformFloatHyperparameter('learning_rate', 1e-4, 1e-2, log=True))
    cs.add_hyperparameter(UniformIntegerHyperparameter('num_units', 16, 128))
    cs.add_hyperparameter(UniformFloatHyperparameter('dropout_rate', 0.0, 0.5))
    return cs

# 6. Configuración y ejecución de BOHB

def run_bohb():
    # Inicia el servidor de nombres
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    # Crea un worker
    worker = MyWorker(nameserver='127.0.0.1', run_id='example1')
    worker.run(background=True)

    # Configura BOHB
    configspace = get_configspace()  # Obtén el espacio de búsqueda
    bohb = BOHB(configspace=configspace, run_id='example1', nameserver='127.0.0.1')

    # Ejecuta la optimización (solo 5 iteraciones para que sea rápido)
    res = bohb.run(n_iterations=5)

    # Detén el servidor y el worker
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    return res

# 7. Generar datos sintéticos
X_train, X_val, y_train, y_val = generate_synthetic_data()

# 8. Ejecutar la optimización
result = run_bohb()

# 9. Mostrar los mejores hiperparámetros encontrados
best_config = result.get_id2config_mapping()[result.get_incumbent_id()]['config']
print("Mejores hiperparámetros encontrados:", best_config)
