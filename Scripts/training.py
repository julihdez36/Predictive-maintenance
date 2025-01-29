# Entrenamiento

###################################################################

# Importación del conjunto de datos

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


X_url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/X_resampled.csv'
y_url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/y_resampled.shape.csv'

X = pd.read_csv(X_url)
y = pd.read_csv(y_url)


print(X.shape[0] == y.shape[0])  











