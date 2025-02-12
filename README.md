# Data-Driven Predictive Maintenance for Electrical Distribution Transformers Using Neural Networks: Bayesian Optimization and Hyperband

## Bayesian Optimization (BOHB) for Predictive Maintenance

The repository is organized as follows:

1. **Scripts:** Contains the execution codes, including preliminary tests with [SMAC3](https://github.com/automl/SMAC3), and [HPBandSter](https://github.com/automl/HpBandSter).
2. **Data:** Contains the working dataset [Dataset of Distribution Transformers](https://data.mendeley.com/datasets/yzyj46xpmy/4)
3. **Deployment:** Contains BOHB training using [SMAC3](https://github.com/automl/SMAC3) package. Also contains training variations.
4. The folders **modelos_soporte** and **modelos** contain the results of different optimized hyperparameter configurations. Of particular interest are those models trained with PCA and autoencoder

## Research Summary

An approach for predictive maintenance based on data from electrical distribution transformers is presented, which explores the use of environmental variables—such as electrical discharges—as an alternative to traditional dissolved gas analysis (DGA) for predicting the probability of transformer failures. The proposal employs a feedforward neural network model, supported by its success in DGA-based models and its universal approximation capability. Since the performance of these models critically depends on the configuration of their hyperparameters, the methodology incorporates an approach that combines Bayesian optimization techniques with Hyperband to automate the simultaneous search for the architecture and hyperparameters of the network (known as Bayesian Optimization and Hyperband [BOHB] \citep{falkner2018bohb}). The treatment of data imbalance, common in predictive maintenance problems, combines the SMOTE oversampling strategy with TOMEK undersampling. Additionally, an autoencoder is used to obtain latent representations, which improves prediction results. The methodology is applied to a dataset of distribution transformers located in the department of Cauca, Colombia, collected by the Compañía Energética de Occidente \citep{bravo2021dataset}. The results obtained are comparatively outstanding in relation to other research.







 
