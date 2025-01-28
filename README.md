## Bayesian Optimization (BOHB) for Predictive Maintenance

The repository is organized as follows:

1. Scripts: Contains the execution codes, including preliminary tests with [SMAC3](https://github.com/automl/SMAC3), and [HPBandSter](https://github.com/automl/HpBandSter).
2. Data: Contains the working dataset [Dataset of Distribution Transformers](https://data.mendeley.com/datasets/yzyj46xpmy/4)

## Research Summary

A methodology for data-driven predictive maintenance of electrical distribution transformers is presented, which explores the use of environmental variables as an alternative to the traditional dissolved gas analysis (DGA) for predicting the probability of transformer failures. The proposal employs a feed-forward neural network model, justified both by its demonstrated success in DGA analysis and its theoretical capacity for universal approximation. Since the performance of these models critically depends on the configuration of their hyperparameters, the methodology incorporates BOHB (Bayesian Optimization and Hyperband) \citep{falkner2018bohb}, an approach that combines Bayesian optimization with Hyperband to simultaneously automate and optimize the network's architecture and hyperparameters. This approach enables efficient and scalable tuning of the model's structure, maximizing its performance. The methodology is applied to a dataset of distribution transformers located in the department of Cauca, Colombia, collected by the Compañía Energética de Occidente, the operator of the region's electrical grid \citep{bravo2021dataset}.







 
