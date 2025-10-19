#################################################################
# Consolidación de datos
##################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


raw19 = 'https://github.com/julihdez36/Predictive-maintenance/raw/refs/heads/main/Data/Dataset_Year_2019.xlsx'
df19 = pd.read_excel(raw19)

raw20 = 'https://github.com/julihdez36/Predictive-maintenance/raw/refs/heads/main/Data/Dataset_Year_2020.xlsx'
df20 = pd.read_excel(raw20)

df20.columns

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

df.isna().sum()

df.client_type.value_counts(normalize = True)


sns.countplot(data = df, x = 'client_type', hue = 'burned_transformers', 
              stat = 'percent', alpha = .5)
plt.xticks(rotation = 45)
plt.show()


###################################################################
# Conjunto de datos para el entrenamiento 1
###################################################################


# df.columns
# df.burning_rate.value_counts()

# df['failed'] = (df['burned_transformers'] + df['burning_rate']) > 0
# df['failed'] = df['failed'].astype(int)
# df.failed.value_counts()

# df.burned_transformers.value_counts()

# df.failed.value_counts()[0] / df.failed.value_counts()[1] # 2.4855072463768115

# df_quemas = df.drop(columns=['eens_kwh','year','failed'])
# df_quemas.columns

# df_entrenamiento_final = df.drop(columns=['burned_transformers','eens_kwh','year','burning_rate'])
# df_entrenamiento_final.columns



########################################################################
# Usuarios
########################################################################

# Separaremos usuarios en: hogares, comercios, industria y oficial


df['client_type'] = df['client_type'].apply(
    lambda x: 'HOUSEHOLD' if 'ESTRATO' in x else x
)

df['client_type'].value_counts()


sns.countplot(data = df, x = 'client_type', hue = 'burned_transformers');
df.columns

# One-hot encoding 
df = pd.get_dummies(df, columns=['client_type'])



########################################################################
# Tipo de instalación
########################################################################

df.columns

# Mapeo de tipos de instalación a inglés
installation_map = {
    'MACRO SIN RED ANTIFRAUDE': 'MACRO OPEN',
    'POSTE': 'POLE-MOUNTED',
    'POSTE RED ANTIFRAUDE': 'POLE WITH ANTI-FRAUD NETWORK',
    'EN H': 'H-FRAME',
    'PAD MOUNTED': 'PAD-MOUNTED',
    'TORRE METALICA': 'METAL TOWER',
    'OTROS': 'OTHERS',
    'CABINA': 'CABINET'
}

# Aplicar el mapeo al DataFrame
df['installation_type_en'] = df['installation_type'].map(installation_map)

# Calcular porcentaje de quemas por tipo de instalación
burn_rate = df.groupby('installation_type_en')['burned_transformers'].mean() * 100
burn_rate = burn_rate.sort_values(ascending=True)

# Crear gráfico de barras horizontales
plt.figure(figsize=(10,6))
sns.barplot(x=burn_rate.values, y=burn_rate.index, palette='viridis')

# Etiquetas y título en inglés
plt.xlabel('Percentage of Burned Transformers (%)', fontsize=12)
plt.ylabel('Installation Type', fontsize=12)
# plt.title('Percentage of Burned Transformers by Installation Type', fontsize=14, fontweight='bold')
plt.xlim((0,38))

# Añadir valores sobre las barras
for i, v in enumerate(burn_rate.values):
    plt.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=10)
plt.grid(linestyle = '--', alpha = .8)
plt.tight_layout()
plt.show()


df.installation_type.value_counts()



df['installation_type'] = df['installation_type'].replace({
    'MACRO SIN RED ANTIFRAUDE': 0, 'POSTE': 1, 'POSTE RED ANTIFRAUDE': 1,
    'EN H': 1, 'PAD MOUNTED': 0, 'TORRE METALICA': 1, 'OTROS': 0, 'CABINA': 0})


df['installation_type'].value_counts() # 1:28306, 0:3440

df = df.drop('installation_type_en', axis = 'columns')

df.columns
########################################################################
# Guardado de datos
########################################################################

df.to_csv('Data\df_entrenamiento_final.csv', index=False)
  









