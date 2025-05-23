##################################################################
# Exploración de variables 
##################################################################


# url = 'https://raw.githubusercontent.com/julihdez36/Predictive-maintenance/refs/heads/main/Data/df.csv'
# df = pd.read_csv(url)

# df.sample(2)
# df.columns

# Visualizaciones de interés

# Configuración del estilo 
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

# Modificar el título y etiquetas
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

# plt.savefig('fallos_año.eps', format='eps', bbox_inches='tight', dpi=300)



df_entrenamiento_final.groupby('installation_type')['failed'].sum()

# Calcular proporciones de fallas y no fallas por tipo de instalación
df_proporciones = (
    df_entrenamiento_final
    .groupby('installation_type')['failed']
    .value_counts(normalize=True)
    .unstack()
)

# Renombrar columnas para mayor claridad
df_proporciones.columns = ['Protected', 'Exposed']

# Convertir a formato largo para usar con Seaborn
df_melted = df_proporciones.reset_index().melt(
    id_vars='installation_type', 
    var_name='Failure Status', 
    value_name='Percentage'
)

# Mapear valores de instalación para etiquetas más claras
df_melted['installation_type'] = df_melted['installation_type'].map({0: 'Protected', 1: 'Exposed'})

# Crear el gráfico de barras apiladas
plt.figure(figsize=(6, 5))
ax = sns.barplot(
    data=df_melted, 
    x='installation_type',  
    y='Percentage', 
    hue='Failure Status', 
    palette=["#4D4D4D", "#BFBFBF"],  # Tonos de gris para diferenciar
    edgecolor="black",
    linewidth=0.5
)

# Modificar el título y etiquetas
plt.title("Distribución de fallas por tipo de instalación", fontsize=12, fontweight="bold")
plt.xlabel("Tipo de instalación", fontsize=10)
plt.ylabel("Proporción", fontsize=10)
plt.ylim(0, 1)  # El eje Y va de 0 a 1 porque representa proporciones

# Agregar etiquetas de porcentaje sobre las barras
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='center', fontsize=9, color="white")

# Modificar la leyenda
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Sin falla", "Falla"], title="Estado del Transformador")

# Añadir cuadrícula
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()
