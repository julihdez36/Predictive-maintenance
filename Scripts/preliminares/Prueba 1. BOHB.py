# Prueba BOHB (Bayesian Optimization and Hyperband)

# Creación de datos sintéticos para clasificación binaria


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Construimos un conjunto de datos de dimensión (500,10) como prueba

X, y = make_classification(
    n_samples=500,   # Número de filas
    n_features=10,   # Número de características
    n_informative=8, # Características relevantes
    n_redundant=2,   # Características redundantes
    random_state=42
)

# Dividimos el conjunto de datos en en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train[:5], y_train[:5]

# Configuración BOHB para prueba rápida

from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.integration.keras import TuneReportCallback
import tensorflow as tf

# Definir espacio de búsqueda reducido
search_space = {
    "num_layers": tune.randint(1, 4),  # Número de capas entre 1 y 3
    "units_per_layer": tune.qrandint(5, 15,q = 1),  # Número de neruonas por capa en rango 1-15
    "learning_rate": tune.loguniform(1e-3, 1e-2),  # Tasa de aprendizaje más restringida
    "batch_size": tune.choice([16, 32]),  # Número de muestras utilizadas en cada paso de entrenamiento.
    "dropout": tune.uniform(0.1, 0.3),  # Fracción de neuronas que se desactivan aleatoriamente
    # durante el entrenamiento para evitar el sobreajuste.10%-30%
    "activation": tune.choice(["relu", "sigmoid"])  # Probamos sólo dos funciones de activación
}


# Creamos una fucnión dinámica para modelar con keras la red con los múltiples parámetros

def build_model(config):
    model = tf.keras.Sequential()
    for _ in range(config["num_layers"]):
        model.add(tf.keras.layers.Dense(config["units_per_layer"], activation=config["activation"]))
        model.add(tf.keras.layers.Dropout(config["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # Para clasificación binaria
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

#

# Configuración del BOHB
# combinamos  Bayesian Optimization con el algoritmo HyperBand para realizar 
# la búsqueda de hiperparámetros de manera eficiente. 

# Primero, utilizamos un modelo bayesiano para explorar el espacio de busqueda

bohb_search = TuneBOHB() 

# Usamos el algoritmo Hyperband que asgina recursos que priorizan
# las configuraciones mas prometedoras basándose en su rendimiento inicial

bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=10, reduction_factor=3)

# Función de entrenamiento

def train_model(config):
    model = build_model(config)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,  # Reducido para prueba rápida
        batch_size=config["batch_size"],
        callbacks=[TuneReportCallback({"accuracy": "val_accuracy"})],
        verbose=0
    )

# Iniciar la búsqueda de hiperparámetros
analysis = tune.run(
    train_model, # Función de entrenamiento (configurada anteriormente)
    config=search_space, # Espacio de busqueda
    search_alg=bohb_search, # Método de búsqueda de hiperparámetros (BOHB)
    scheduler=bohb_scheduler, # Estrategia de asignación de recursos (HyperBand)
    num_samples=10,  # Número de configuraciones a probar
    metric="accuracy", # Métrica objetivo para optimizar
    mode="max"    # Objetivo: Maximizar la métrica
)

# Evaluar el mejor conjnuto de hiperparametros

print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))



