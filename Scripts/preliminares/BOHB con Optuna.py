import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback

# 1. Generar datos sintéticos (igual que antes)
def generate_synthetic_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

# Cargar datos
X_train, X_val, y_train, y_val = generate_synthetic_data()

# 2. Definir el modelo (similar al original, pero usando los hiperparámetros de Optuna)
def create_model(trial):
    # Definir hiperparámetros con Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_units = trial.suggest_int("num_units", 16, 128)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])

    # Construir modelo
    model = Sequential()
    model.add(Dense(num_units, input_dim=20, activation=activation))
    for _ in range(num_layers - 1):
        model.add(Dense(num_units, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Callback para reportar métricas a Optuna (necesario para el pruning)
class OptunaCallback(Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        # Reportar métrica intermedia para pruning
        self.trial.report(logs["val_loss"], step=self.epoch)
        # Detener el entrenamiento si Optuna sugiere pruning
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

# 4. Función objetivo
def objective(trial):
    # Crear modelo y definir hiperparámetros
    model = create_model(trial)
    
    # Número máximo de épocas (budget)
    max_epochs = 20
    
    # Entrenar el modelo con callback para pruning
    history = model.fit(
        X_train,
        y_train,
        epochs=max_epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[OptunaCallback(trial)]
    )
    
    # Devolver la mejor pérdida de validación
    return min(history.history["val_loss"])

# 5. Configurar y ejecutar el estudio de Optuna
def run_optuna():
    # Definir pruner y sampler
    pruner = HyperbandPruner(min_resource=1, max_resource=20, reduction_factor=3)
    sampler = TPESampler(seed=42)  # Para reproducibilidad
    
    # Crear estudio
    study = optuna.create_study(
        direction="minimize",  # Minimizar la pérdida de validación
        sampler=sampler,
        pruner=pruner,
    )
    
    # Ejecutar optimización
    study.optimize(objective, n_trials=50)
    
    # Mostrar resultados
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print(f"Mejor val_loss: {study.best_value:.4f}")
    
    return study

# 6. Ejecutar
if __name__ == "__main__":
    study = run_optuna()