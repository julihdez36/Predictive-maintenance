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

df.isna().sum()

#################################################################
# Conjunto de datos concatenado
##################################################################

#df.to_csv('Data\df.csv', index=False)  # Guarda el DataFrame en un archivo CSV, sin incluir el índice


###################################################################
# Conjunto de datos para el entrenamiento 1
###################################################################


df.columns
df.burning_rate.value_counts()

df['failed'] = (df['burned_transformers'] + df['burning_rate']) > 0
df['failed'] = df['failed'].astype(int)
df.failed.value_counts()

df.burned_transformers.value_counts()

df.failed.value_counts()[0] / df.failed.value_counts()[1] # 2.4855072463768115

df_quemas = df.drop(columns=['eens_kwh','year','failed'])
df_quemas.columns

df_entrenamiento_final = df.drop(columns=['burned_transformers','eens_kwh','year','burning_rate'])
df_entrenamiento_final.columns

########################################################################
# Usuarios
########################################################################

# Separaremos usuarios en: hogares, comercios, industria y oficial


df_entrenamiento_final['client_type'] = df_entrenamiento_final['client_type'].apply(
    lambda x: 'HOUSEHOLD' if 'STRATUM' in x else x
)

df_entrenamiento_final ['client_type'].value_counts()

# One-hot encoding 
df_entrenamiento_final = pd.get_dummies(df_entrenamiento_final, columns=['client_type'])


# Solo con quemas


df_quemas['client_type'] = df_quemas['client_type'].apply(
    lambda x: 'HOUSEHOLD' if 'STRATUM' in x else x
)

df_quemas['client_type'].value_counts()

# One-hot encoding 
df_quemas = pd.get_dummies(df_quemas, columns=['client_type'])

df_quemas.client_type_OFFICIAL.sum() # 132
df_quemas.client_type_INDUSTRIAL.sum() # 238
df_quemas.client_type_COMMERCIAL.sum() #676
df_quemas.client_type_HOUSEHOLD.sum() #30700

########################################################################
# Tipo de instalación
########################################################################


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

# Con quemas ####

df_quemas.installation_type.value_counts()

df_quemas['installation_type'] = df_quemas['installation_type'].replace({
    'POLE WITH ANTI-FRAU NET': 'POLE WITH ANTI-FRAUD NET'
})

df_quemas['installation_type'] = df_quemas['installation_type'].replace({
        'POLE': 1, # EXPOSED
        'POLE WITH ANTI-FRAUD NET': 1,
        'TORRE METALICA': 1,
        'EN H': 1,
        'CABINA': 0, # PROTECTED
        'PAD MOUNTED': 0, 
        'MACRO WITHOUT ANTI-FRAUD NET': 0,
        'OTROS': 0})
        
       
df_quemas['installation_type'].value_counts() # 1:28306, 0:3440

df_quemas.columns

df_quemas['burned_transformers'].value_counts() # 0:30310, 1:1436

df_quemas.groupby('installation_type')['burned_transformers'].sum()


########################################################################
# Tipado de variables
########################################################################


df_entrenamiento_final['network_km_lt'] = df_entrenamiento_final['network_km_lt'].astype(str).str.replace(',', '', regex=True)
df_entrenamiento_final['network_km_lt'] = df_entrenamiento_final['network_km_lt'].astype(float)

df_entrenamiento_final.isna().sum()
df_entrenamiento_final.info()


# Con quemas

df_quemas['network_km_lt'] = df_quemas['network_km_lt'].astype(str).str.replace(',', '', regex=True)
df_quemas['network_km_lt'] = df_quemas['network_km_lt'].astype(float)

df_quemas.isna().sum()
df_quemas.info()







df_entrenamiento_final['failed'].value_counts()
# 0    22638
# 1     9108

df_quemas['burned_transformers'].value_counts() 

# 0:30310
# 1:1436


df_entrenamiento_final.shape
df_quemas.shape

########################################################################
# Guardado de datos
########################################################################

df_entrenamiento_final.to_csv('Data\df_entrenamiento_final.csv', index=False)
  
df_quemas.to_csv('Data\df_quemas.csv', index=False)









