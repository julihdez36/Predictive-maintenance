# Data-driven predictive maintenance of distribution transformers: class imbalance and Bayesian optimization

The repository is organized as follows:

1. **Data:** Contains the working dataset [Dataset of Distribution Transformers](https://data.mendeley.com/datasets/yzyj46xpmy/4)
2. **Deployment:** Contains BOHB training using [SMAC3](https://github.com/automl/SMAC3) package. Also contains training variations.

## Abstract

An approach for data-driven predictive maintenance of electrical distribution transformers is presented, where the probability of failure is modeled as a function of operational and meteorological variables. Since failures are rare events, predictive maintenance tasks often face class imbalance scenarios. To address this, a hybrid strategy is explored, combining data-level treatment (SMOTE-Tomek) and algorithm-level adjustments (focal loss). Additionally, the impact of hyperparameter optimization via BOHB (Bayesian Optimization with HyperBand) on neural networks is evaluated.
Results show that proper handling of class imbalance is crucial for improving model predictive performance, while hyperparameter optimization offers marginal improvements at the cost of high computational overhead. The methodology is applied to a dataset of distribution transformers located in the Cauca department, Colombia, collected by Compañía Energética de Occidente. The achieved results are comparable to those from specialized models based on chemical testing (DGA), demonstrating the potential of low-cost, readily available data-driven approaches.

## Keywords:

Predictive maintenance, Class imbalance, SMOTE-Tomek, Focal loss, BOHB (Bayesian Optimization with HyperBand), Power distribution transformers.

---

## Resumen: 

Se presenta un enfoque para el mantenimiento predictivo basado en datos de transformadores de distribución eléctrica, en el que se considera la probabilidad de fallos como una función dependiente de variables operativas y meteorológicas. Dado que los fallos son eventos atípicos, es común que las tareas de mantenimiento predictivo enfrenten contextos de desbalance de clases. Para ello, se explora una estrategia híbrida que combina un tratamiento a nivel de datos (SMOTE-Tomek) y a nivel del algoritmo (focal loss). Además, se evalúa el impacto de la optimización de hiperparámetros mediante BOHB (Bayesian Optimization with HyperBand) en redes neuronales. Los resultados muestran que un tratamiento adecuado del desbalance es determinante para mejorar la capacidad predictiva de los modelos, mientras que la optimización de hiperparámetros ofrece mejoras marginales a costa de un alto costo computacional. La metodología se aplica a un conjunto de datos de transformadores de distribución ubicados en el departamento del Cauca, Colombia, recopilados por la Compañía Energética de Occidente \citep{bravo2021dataset}. Los resultados obtenidos son comparables con los alcanzados por modelos especializados basados en pruebas químicas (DGA), lo que refleja el potencial de enfoques basados en datos de bajo costo y fácil disponibilidad.

## Palabras clave: 

Mantenimiento predictivo, Desbalance de clases, SMOTE-Tomek, Focal loss, BOHB (Optimización Bayesiana con HyperBand), Transformadores de distribución eléctrica.



 
