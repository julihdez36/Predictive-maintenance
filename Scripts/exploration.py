#################################################################
##### Exploration 

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


##################################################################
# Exploración de variables 

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df.csv'
df = pd.read_csv(url)

df.sample(2)
df.columns

##################################################################
# Variables dependientes

df.burned_transformers.value_counts()

df['burning_rate'].describe()
##################################################################

# Visualizaciones de interés

# Configuración del estilo para gráficos IEEE
plt.rcParams.update({
    "text.usetex": True,  # Usar LaTeX para renderizar texto
    "font.family": "serif",  # Usar una fuente serif (como Times New Roman)
    "font.serif": ["Times New Roman"],  # Especificar Times New Roman
    "font.size": 10,  # Tamaño de la fuente
    "axes.titlesize": 10,  # Tamaño del título de los ejes
    "axes.labelsize": 10,  # Tamaño de las etiquetas de los ejes
    "xtick.labelsize": 8,  # Tamaño de las etiquetas del eje X
    "ytick.labelsize": 8,  # Tamaño de las etiquetas del eje Y
    "legend.fontsize": 8,  # Tamaño de la leyenda
    "figure.titlesize": 10,  # Tamaño del título de la figura
    "lines.linewidth": 1.5,  # Grosor de las líneas
    "lines.markersize": 6,  # Tamaño de los marcadores
    "grid.color": "gray",  # Color de la cuadrícula
    "grid.linestyle": ":",  # Estilo de la cuadrícula
    "grid.linewidth": 0.5,  # Grosor de la cuadrícula
})

# Fallos de trasnformadores por area (0:rural, 1: urbana)

df.burned_transformers.sum() # 1436 fallos totales



###############################################
# Cantidad de Transformadores Quemados por Zona

tabla_fallas = df.groupby('location')['burned_transformers'].sum()     
# 1384 en zona rural, 52 en zonas urbanas
plt.figure(figsize=(8, 6))  # Ajustar tamaño
ax = sns.barplot(
    x=tabla_fallas.index, 
    y=tabla_fallas.values, 
    palette="Set2", 
    edgecolor="black",
    linewidth=0.5
)

plt.title("Cantidad de Transformadores Quemados por Zona", fontsize=12, fontweight="bold")
plt.xlabel("Zona", fontsize=10)
plt.ylabel("Transformadores Quemados", fontsize=10)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()

plt.show()



# Guardar la figura en formato adecuado para IEEE
plt.savefig('transformadores_quemados.eps', format='eps', bbox_inches='tight', dpi=300)

###################################################################
# Variables que considero omitir


df.criticality_ceramics # Criticalidad por estudios previos
df.eens_kwh # riesgo que implica cesar la prestación del servicio


###################################################################

# Variables a ajustar

df.client_type.value_counts() # Ajustar: hogars y empresas

# Tipo de instalación
# Podemos explorar una combinación one-hot para variables categóricas

df.installation_type.value_counts() # Pensar cómo ajustar


###################################################################

# Variable dependiente del modelo
# Problema de clasificación binario

df['burned_transformers'].value_counts()






