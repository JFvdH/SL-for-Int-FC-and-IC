# Supervised Learning for Integrated Forecasting and Inventory Control

This repository contains the code used for the numerical experiments in the "Supervised Learning for Integrated Forecasting and Inventory Control" paper. Each folder contains code for the inventory management problem it is named after. 

The folders 'Stationary dual sourcing', 'Stationary lost sales' and 'Stationary perishable goods' contain the code used for the experiments under stationary demand, as described in Section 6 of the paper. Each of these folders contains the following files:
- 'main.py' can be used to change parameter settings and start the experiments.
- 'hyperparams.py' contains code used for hyperparameter optimization.
- 'generateData.py' is used to generate training data according to the specified distribution.
- 'train.py' uses generated data to train a LightGBM model using the specified hyperparameters.
- 'evalModel.py' evaluates a trained LightGBM model using simulation runs.
- 'utility.py' contains a utility function that can be used to monitor training and tuning progress.

The folder 'Lumpy lost sales' contains the code used for the experiments under lumpy demand, as described in Appendix B of the paper. It contains the following files:
- 'main.py' can be used to change parameter settings and start the experiments.
- 'hyperparams.py' contains code used for hyperparameter optimization.
- 'createData.py' is used to generate time series training data with a specified length, number of features and (average) demand interval.
- `augmentData.py' is used to augment the created training data in line with the given (hyper)parameters.
- 'train.py' uses generated data to train a LightGBM model using the specified hyperparameters.
- 'evalModel.py' evaluates a trained LightGBM model using simulation runs.
- 'utility.py' contains a utility function that can be used to monitor training and tuning progress.