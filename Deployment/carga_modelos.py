import tensorflow as tf
import joblib
import json

# Cargar el modelo
modelo = tf.keras.models.load_model('modelo_fitted.h5')

# Cargar el escalador
scaler = joblib.load('scaler.pkl')

# Cargar los hiperparámetros
with open('hiperparametros.json', 'r') as f:
    best_config = json.load(f)

print("Hiperparámetros óptimos:")
print(best_config)
