# Optimización para un local

import ConfigSpace as CS
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Crear el conjunto de datos sintéticos
# Optamos por un dataset más pequeño para pruebas locales
X, y = make_classification(
    n_samples=200,       # Número de filas reducido para pruebas rápidas
    n_features=10,       # Número de características
    n_informative=8,     # Características relevantes
    n_redundant=2,       # Características redundantes
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el espacio de búsqueda con ConfigSpace
# Reducimos los rangos para minimizar el número de configuraciones a explorar

config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("num_layers", lower=1, upper=2))  # Reducimos las capas a 1 o 2
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("units_per_layer", lower=5, upper=10))  # Menos neuronas por capa
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("learning_rate", lower=1e-3, upper=5e-3, log=True))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("batch_size", [16, 32]))  # Dos opciones de batch size
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("dropout", lower=0.1, upper=0.2))  # Dropout más pequeño
config_space.add_hyperparameter(CS.CategoricalHyperparameter("activation", ["relu"]))  # Solo 'relu' como activación

# Crear una clase Worker para entrenar el modelo
class KerasWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, config, budget, **kwargs):
        # Crear el modelo secuencial con los hiperparámetros del espacio de búsqueda
        model = tf.keras.Sequential()
        for _ in range(config["num_layers"]):
            model.add(tf.keras.layers.Dense(config["units_per_layer"], activation=config["activation"]))
            model.add(tf.keras.layers.Dropout(config["dropout"]))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        # Agregar early stopping para detener entrenamientos poco prometedores
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=int(budget),  # El presupuesto actúa como número de épocas
            batch_size=config["batch_size"],
            callbacks=[early_stopping],  # Detener temprano si no mejora
            verbose=0  # Sin salida en consola para mantener el enfoque en la optimización
        )
        
        # Retornar la última precisión de validación
        val_accuracy = history.history["val_accuracy"][-1]
        return {"loss": 1 - val_accuracy, "info": {"val_accuracy": val_accuracy}}

# Configurar BOHB para optimización
# Reducimos el presupuesto máximo y el número de iteraciones para tiempos más rápidos

bohb = BOHB(
    configspace=config_space,
    run_id="keras_bohb_test",
    min_budget=1,  # Presupuesto mínimo en épocas
    max_budget=5   # Presupuesto máximo reducido para ahorrar tiempo
)

# Ejecutar la optimización
result = bohb.run(
    n_iterations=5,  # Menos iteraciones para reducir tiempo de ejecución
    min_n_workers=1  # Se ejecuta de forma secuencial en una máquina local
)

# Obtener la mejor configuración encontrada
best_config = result.get_id2config_mapping()[result.get_incumbent_id()]["config"]
print("Best configuration:", best_config)

# Finalizar BOHB
bohb.shutdown()
