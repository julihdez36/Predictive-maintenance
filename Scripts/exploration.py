# Exploration

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

raw19 = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/Dataset_Year_2019.csv'
df19 = pd.read_csv(raw19)

raw20 = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/Dataset_Year_2020.csv'
df20 = pd.read_csv(raw20)

new_names = ['location', 'power', 'self_protection', 
 'avg_earth_ddt', 'max_earth_ddt', 
 'burning_rate', 'criticality_ceramics', 
 'removable_connectors', 'client_type', 
 'num_users', 'eens_kwh', 
 'installation_type', 'air_network', 
 'circuit_queue', 'network_km_lt', 
 'burned_transformers']

df19.columns, df20.columns = new_names, new_names

df19['year'] = 2019
df20['year']= 2020

print(f'Dimesión de los df: {df19.shape, df20.shape}')

df19.info()

df19.burned_transformers.value_counts()

print(df19.burned_transformers.value_counts())

print(df20.burned_transformers.value_counts())

df = pd.concat([df19,df20], ignore_index= True)
print(df.shape)
df.sample(3)


# Carga final del conjunto de datos

df.to_csv('Data\df.csv', index=False)  # Guarda el DataFrame en un archivo CSV, sin incluir el índice


# Visualización del los transformadores

sns.set_theme(style="whitegrid")  # Cambia a 'darkgrid', 'white', 'dark', etc., según prefieras.

# Crear el gráfico
plt.figure(figsize=(8, 6))  # Ajustar el tamaño de la figura
ax = sns.countplot(data=df, x='burned_transformers', hue = 'burned_transformers')  # Cambia la paleta según el estilo deseado

# Etiquetas en el eje x e y
ax.set_xlabel('Transformadores Quemados', fontsize=14, labelpad=10)
ax.set_ylabel('Cantidad', fontsize=14, labelpad=10)

# Título
ax.set_title('Distribución de Transformadores Quemados', fontsize=16, fontweight='bold', pad=20)

# Añadir etiquetas de conteo sobre las barras
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=12, padding=3)

# Ajustar las etiquetas del eje x si son muy largas
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Ajustar espacio del gráfico para que el título no quede pegado
plt.tight_layout()

# Mostrar el gráfico
plt.show()





