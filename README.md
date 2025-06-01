# Data-driven predictive maintenance of distribution transformers: class imbalance and Bayesian optimization

The repository is organized as follows:

1. **Data:** Contains the working dataset [Dataset of Distribution Transformers](https://data.mendeley.com/datasets/yzyj46xpmy/4)
2. **Deployment:** Contains BOHB training using [SMAC3](https://github.com/automl/SMAC3) package. Also contains training variations.

## Abstract

An approach for predictive maintenance based on data from electrical distribution transformers is presented, where the probability of failures is considered a function dependent on easily accessible features such as load, transformer specifications, and meteorological variables. Given that component failures are rare events, predictive maintenance strategies are typically developed in the context of class imbalance. To address this issue, this work employs a hybrid strategy that combines data-level techniques (SMOTE-Tomek) and algorithm-level techniques (focal loss). To enhance the failure detection rate, a probabilistic hyperparameter optimization technique known as BOHB is explored. The results demonstrate that proper handling of class imbalance is crucial for improving the predictive performance of models, while hyperparameter optimization yields marginal improvements at a high computational cost. The methodology is applied to a dataset of distribution transformers located in the Cauca department, Colombia, collected by the Compañía Energética de Occidente. The obtained results are comparable to those achieved by specialized models based on chemical tests (DGA), highlighting the potential of approaches relying on low-cost and readily available data.

## Keywords:

Predictive maintenance, Class imbalance, SMOTE-Tomek, Focal loss, BOHB (Bayesian Optimization with HyperBand), Power distribution transformers.

---

## Resumen: 

Se presenta un enfoque para el mantenimiento predictivo basado en datos de transformadores de distribución eléctrica, en el que se considera la probabilidad de fallos como una función dependiente de caracteristicas fácilmente accesibles como la carga, las especificaciones del transformador y las variables meteorológicas. Dado que los fallos en componentes son eventos atípicos, las estrategias de mantenimiento predictivo suelen desarrollarse en contextos de desbalance de clase. Para tratar esta situación, en este trabajo se despliega una estrategia híbrida que combina técnias al nivel de datos (SMOTE-Tomek) y a nivel del algoritmo (focal loss). Con el propósito de mejorar la tasa de descubrimiento de fallos, se explora una técnica de optimización probabilistica de hiperparámetros conocida como BOHB. Los resultados muestran que un tratamiento adecuado del desbalance es determinante para mejorar la capacidad predictiva de los modelos, mientras que la optimización de hiperparámetros ofrece mejoras marginales a costa de un alto costo computacional. La metodología se aplica a un conjunto de datos de transformadores de distribución ubicados en el departamento del Cauca, Colombia, recopilados por la Compañía Energética de Occidente. Los resultados obtenidos son comparables con los alcanzados por modelos especializados basados en pruebas químicas (DGA), lo que refleja el potencial de enfoques basados en datos de bajo costo y fácil disponibilidad.

## Palabras clave: 

Mantenimiento predictivo, Desbalance de clases, SMOTE-Tomek, Focal loss, BOHB (Optimización Bayesiana con HyperBand), Transformadores de distribución eléctrica.



 
