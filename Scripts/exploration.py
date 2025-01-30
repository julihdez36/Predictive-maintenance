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


# Trasnformadores quemados por año

plt.figure(figsize=(8, 6)) 
ax = sns.countplot(
    data= df,
    x= 'year', 
    hue= 'burned_transformers',  
    palette="Greys", 
    edgecolor="black",
    linewidth=0.5
)

# Modificar el título y etiquetas
plt.title("Cantidad de Transformadores Quemados por año", fontsize=12, fontweight="bold")
plt.xlabel("Año", fontsize=10)
plt.ylabel("Transformadores Quemados", fontsize=10)

for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=8)

handles, labels = ax.get_legend_handles_labels()
new_labels = ['No fallo', 'Fallo']  # Nuevas etiquetas para la leyenda
ax.legend(handles, new_labels, title="Estado del Transformador")

ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()

plt.savefig('fallos_año.eps', format='eps', bbox_inches='tight', dpi=300)

df.burned_transformers.value_counts()


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

# Saving .eps
# plt.savefig('transformadores_quemados.eps', format='eps', bbox_inches='tight', dpi=300)

###################################################################
# Variables que considero omitir


df.criticality_ceramics # Criticalidad por estudios previos
df.eens_kwh # riesgo que implica cesar la prestación del servicio

df_final = df.drop(columns = ['criticality_ceramics','eens_kwh'])
df_final.columns

###################################################################

# Variables a ajustar

# Diferenciaremos hogares, comercios, industria y oficial

df_final['client_type'] = df['client_type'].apply(
    lambda x: 'HOUSEHOLD' if 'STRATUM' in x else x
)

# One-hot encoding 
df_final = pd.get_dummies(df_final, columns=['client_type'])

df_final.columns

# Tipo de instalación
# Podemos explorar una combinación one-hot para variables categóricas

df_final.installation_type.value_counts()


df_final['installation_type'] = df['installation_type'].replace({
    'POLE WITH ANTI-FRAU NET': 'POLE WITH ANTI-FRAUD NET'
})

# Agrupar categorías poco frecuentes en "OTROS"


df_final['installation_type'] = df_final['installation_type'].replace({
    'CABINA': 'OTROS',
    'TORRE METALICA': 'OTROS',
    'PAD MOUNTED': 'OTROS'
})


# Aplicar One-Hot Encoding

df_final = pd.get_dummies(df_final, columns=['installation_type'], prefix='installation')


# Tipado del modelo

df_final.info()


df_final['network_km_lt'] = df_final['network_km_lt'].str.replace(',', '').astype(float)


###################################################################

# Variable dependiente del modelo
# Problema de clasificación binario

df_final['burned_transformers'].value_counts()

df_final.to_csv('Data\df_entrenamiento.csv', index=False)  



###################################################################

# división y balanceo de datos

# SMOTE (sobremuestreo de la clase minoritaria)
# Tomek Links (submuestreo de la clase mayoritaria)


from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
from sklearn.model_selection import train_test_split

df_final = pd.read_csv('Data\df_entrenamiento.csv')


# Separar características (X) y etiqueta (y)
X = df_final.drop(columns=['burned_transformers'])  # Ajusta el nombre de la columna si es necesario
y = df_final['burned_transformers']

# Verificar el desbalance de clases
print("Distribución de clases antes del remuestreo:", Counter(y))


# Aplicar SMOTE + Tomek Links

# Demenos ajustas esto porque me está produciendo mas datos de los reales
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

# Verificar la nueva distribución de clases
print("Distribución de clases después de SMOTE + Tomek Links:", Counter(y_resampled))

X_resampled.shape
y_resampled.shape


X_resampled.to_csv('Data\X_resampled.csv', index=False)  
y_resampled.to_csv('Data\y_resampled.shape.csv', index=False)  



import pandas as pd
from imblearn.combine import SMOTETomek
from collections import Counter
from sklearn.model_selection import train_test_split

df_final = pd.read_csv('Data/df_entrenamiento.csv')

# Separar características (X) y etiqueta (y)
X = df_final.drop(columns=['burned_transformers'])  
y = df_final['burned_transformers']

# Verificar el desbalance de clases
print("Distribución de clases antes del remuestreo:", Counter(y))

# Dividir en conjunto de entrenamiento y prueba antes del balanceo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE + Tomek Links solo en el conjunto de entrenamiento
smote_tomek = SMOTETomek(sampling_strategy=.5, random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Verificar la nueva distribución de clases
print("Distribución de clases después de SMOTE + Tomek Links:", Counter(y_resampled))

# Guardar los datos balanceados
X_resampled.to_csv('Data/X_resampled.csv', index=False)
y_resampled.to_csv('Data/y_resampled.csv', index=False)

X_test.to_csv('Data/X_test.csv', index=False)
y_test.to_csv('Data/y_test.csv', index=False)





