# Supervised Learning for Integrated Forecasting and Inventory Control: An Application to Lost Sales, Perishable Goods and Dual Sourcing

This repository contains the code used for the numerical experiments with stationary demand in the "Supervised Learning for Integrated Forecasting and Inventory Control: An Application to Lost Sales, Perishable Goods and Dual Sourcing" paper. Each folder contains code for the inventory management problem it is named after. The following files are included in each of the folders:
- 'main.py' can be used to change parameter settings and start the experiments.
- 'hyperparams.py' contains code used for hyperparameter optimization.
- 'generateData.py' is used to generate training data according to the specified distribution.
- 'train.py' uses generated data to train a LightGBM model using the specified hyperparameters.
- 'evalModel.py' evaluates a trained LightGBM model using simulation runs.
- 'utility.py' contains a utility function that can be used to monitor training and tuning progress.
