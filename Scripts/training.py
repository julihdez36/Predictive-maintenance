# Entrenamiento

###################################################################

# Importación del conjunto de datos

import pandas as pd

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento.csv'
df_final = pd.read_csv(url)

df_final.info()


# división y balanceo de datos

# SMOTE (sobremuestreo de la clase minoritaria)
# Tomek Links (submuestreo de la clase mayoritaria)


from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.model_selection import train_test_split

# Separar características (X) y etiqueta (y)
X = df_final.drop(columns=['burned_transformers'])  # Ajusta el nombre de la columna si es necesario
y = df_final['burned_transformers']

# Verificar el desbalance de clases
print("Distribución de clases antes del remuestreo:", Counter(y))


# Aplicar SMOTE + Tomek Links
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Verificar la nueva distribución de clases
print("Distribución de clases después de SMOTE + Tomek Links:", Counter(y_resampled))

X_resampled.shape
y_resampled.shape
