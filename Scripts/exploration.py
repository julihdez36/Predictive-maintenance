#################################################################
# Consolidación de datos
##################################################################

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

print(f'Dimesión de los df: {df19.shape, df20.shape}') #((15873, 17), (15873, 17))

df = pd.concat([df19,df20], ignore_index= True)
print(df.shape) # (31746, 17)
df.sample(3)

#################################################################
# Conjunto de datos concatenado
##################################################################

df.to_csv('Data\df.csv', index=False)  # Guarda el DataFrame en un archivo CSV, sin incluir el índice


###################################################################
# Conjunto de datos para el entrenamiento
###################################################################

df.columns
df.burning_rate.value_counts()

df['failed'] = (df['burned_transformers'] + df['burning_rate']) > 0
df['failed'] = df['failed'].astype(int)
df.failed.value_counts()

df.failed.value_counts()[0] / df.failed.value_counts()[1] # 2.4855072463768115


df.drop(columns=['burned_transformers','eens_kwh','year','burning_rate'])


df_entrenamiento_final = df.drop(columns=['burned_transformers','eens_kwh','year','burning_rate'])
df_entrenamiento_final.columns


# Separaremos usuarios en: hogares, comercios, industria y oficial

df['client_type'].value_counts()

df_entrenamiento_final['client_type'] = df_entrenamiento_final['client_type'].apply(
    lambda x: 'HOUSEHOLD' if 'STRATUM' in x else x
)

# One-hot encoding 
df_entrenamiento_final = pd.get_dummies(df_entrenamiento_final, columns=['client_type'])

df_entrenamiento_final.client_type_OFFICIAL.sum() # 132
df_entrenamiento_final.client_type_INDUSTRIAL.sum() # 238
df_entrenamiento_final.client_type_COMMERCIAL.sum() #676
df_entrenamiento_final.client_type_HOUSEHOLD.sum() #30700

# Tipo de instalación

df_entrenamiento_final.installation_type.value_counts()

df_entrenamiento_final['installation_type'] = df_entrenamiento_final['installation_type'].replace({
    'POLE WITH ANTI-FRAU NET': 'POLE WITH ANTI-FRAUD NET'
})

df_entrenamiento_final['installation_type'] = df_entrenamiento_final['installation_type'].replace({
        'POLE': 1, # EXPOSED
        'POLE WITH ANTI-FRAUD NET': 1,
        'TORRE METALICA': 1,
        'EN H': 1,
        'CABINA': 0, # PROTECTED
        'PAD MOUNTED': 0, 
        'MACRO WITHOUT ANTI-FRAUD NET': 0,
        'OTROS': 0})
        
        
df_entrenamiento_final['installation_type'].value_counts() # 1:28306, 0:3440
df_entrenamiento_final['failed'].value_counts() # 0:22638, 1:9108

df_entrenamiento_final.groupby('installation_type')['failed'].sum()


# Tipado de variables

df_entrenamiento_final['network_km_lt'] = df_entrenamiento_final['network_km_lt'].str.replace(',', '').astype(float)


df_entrenamiento_final.info()

###################################################################

# Variable dependiente del modelo
# Problema de clasificación binario

df_entrenamiento_final['failed'].value_counts()


# Aquí voy!
df_entrenamiento_final.to_csv('Data\df_entrenamiento_final.csv', index=False)  


##################################################################
# Exploración de variables 

url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df_entrenamiento_final.csv'
df_entrenamiento_final = pd.read_csv(url)

df.sample(2)
df.columns
df.info()


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

df_entrenamiento_final.columns

# Número de fallas

plt.figure(figsize= (8,6))
ax = sns.countplot(
    data= df_entrenamiento_final,
    x= 'failed', 
    palette="Greys", 
    edgecolor="black",
    linewidth=0.5
)

plt.title("Fallas en transformadores por ubicación", fontsize=12, fontweight="bold")
plt.xlabel("Ubicación", fontsize=10)
plt.ylabel("Fallas en transformadores", fontsize=10)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

handles, labels = ax.get_legend_handles_labels()
new_labels = ['No fallo', 'Fallo']  # Nuevas etiquetas para la leyenda
ax.legend(handles, new_labels, title="Estado del Transformador")

ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()

# Cantidad de fallas en transformadores por zona

plt.figure(figsize=(8, 6)) 
ax = sns.countplot(
    data= df_entrenamiento_final,
    x= 'location', 
    hue= 'failed',  
    palette="Greys", 
    edgecolor="black",
    linewidth=0.5
)

# plt.title("Transformer failures by location", fontsize=12, fontweight="bold")
plt.xlabel("Location", fontsize=10)
plt.ylabel("Transformer faults", fontsize=10)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

handles, labels = ax.get_legend_handles_labels()
new_labels = ['No Failure', 'Failure']  # Nuevas etiquetas para la leyenda
ax.legend(handles, new_labels, title="Transformer Status")

ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()

plt.savefig('fallos_region.eps', format='eps', bbox_inches='tight', dpi=300)



df_entrenamiento_final.info()

df_entrenamiento_final.failed.value_counts(normalize = True) *100
df_entrenamiento_final.failed.value_counts() 



