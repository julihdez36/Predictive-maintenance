import ConfigSpace as CS
from hpbandster.optimizers import BOHB
from hpbandster.core.worker import Worker
from ray.tune.schedulers import HyperBandForBOHB
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Crear el conjunto de datos sintéticos
X, y = make_classification(
    n_samples=500,   # Número de filas
    n_features=10,   # Número de características
    n_informative=8, # Características relevantes
    n_redundant=2,   # Características redundantes
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el espacio de búsqueda con ConfigSpace
config_space = CS.ConfigurationSpace()
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("num_layers", lower=1, upper=3))
config_space.add_hyperparameter(CS.UniformIntegerHyperparameter("units_per_layer", lower=5, upper=15))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("learning_rate", lower=1e-3, upper=1e-2, log=True))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("batch_size", [16, 32]))
config_space.add_hyperparameter(CS.UniformFloatHyperparameter("dropout", lower=0.1, upper=0.3))
config_space.add_hyperparameter(CS.CategoricalHyperparameter("activation", ["relu", "sigmoid"]))

# Crear una clase Worker para entrenar el modelo
class KerasWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, config, budget, **kwargs):
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
        # Entrenar el modelo
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=int(budget),  # El presupuesto actúa como número de épocas
            batch_size=config["batch_size"],
            verbose=0
        )
        # Retornar la última precisión de validación
        val_accuracy = history.history["val_accuracy"][-1]
        return {"loss": 1 - val_accuracy, "info": {"val_accuracy": val_accuracy}}

# Configurar BOHB
bohb = BOHB(
    configspace=config_space,
    run_id="keras_bohb_test",
    min_budget=1,  # Presupuesto mínimo (en épocas)
    max_budget=10  # Presupuesto máximo (en épocas)
)

# Usar HyperBand como programador de recursos
bohb_scheduler = HyperBandForBOHB(
    time_attr="training_iteration", max_t=10, reduction_factor=3
)

# Ejecutar la optimización
result = bohb.run(
    n_iterations=10,  # Número de iteraciones de optimización
    min_n_workers=1
)

# Evaluar los mejores parámetros
best_config = result.get_id2config_mapping()[result.get_incumbent_id()]["config"]
print("Best configuration:", best_config)
