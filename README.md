# Supervised Learning for Integrated Forecasting and Inventory Control

## Description

This repository contains code used for the numerical experiments in the *Supervised Learning for Integrated Forecasting and Inventory Control* paper by Joost F. van der Haar, Arnoud P. Wellens, Robert N. Boute and Rob J.I. Basten. A preprint of the paper can be found [here](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4406486). 

Abstract:
> We explore the use of supervised learning with custom loss functions for multi-period inventory control with feature-driven demand. This method directly considers feature information such as promotions and trends to make periodic order decisions, does not require distributional assumptions on demand, and is sample efficient. The application of supervised learning in inventory control has thus far been limited to problems for which the optimal policy structure is known and takes the form of a simple decision rule, such as the newsvendor problem. We present an approximation approach to expand its use to inventory problems where the optimal policy structure is unknown. We test our approach on lost sales, perishable goods, and dual-sourcing inventory problems. It performs on par with state-of-the-art heuristics under stationary demand. It outperforms them for non-stationary perishable goods settings where demand is driven by features, and for non-stationary lost sales and dual-sourcing settings where demand is smooth and feature-driven.

## Overview

Each folder contains code for the inventory management problem it is named after. 

The folders *Stationary dual sourcing/*, *Stationary lost sales/* and *Stationary perishable goods/* contain the code used for the experiments under stationary demand, as described in Section 6 of the paper. Each of these folders contains the following files:
- *main.py* can be used to change parameter settings and start the experiments.
- *hyperparams.py* contains code used for hyperparameter optimization.
- *generateData.py* is used to generate training data according to the specified distribution.
- *train.py* uses generated data to train a LightGBM model using the specified hyperparameters.
- *evalModel.py* evaluates a trained LightGBM model using simulation runs.
- *utility.py* contains a utility function that can be used to monitor training and tuning progress.

The folders *Non-stationary dual sourcing/*, *Non-stationary lost sales/* and *Non-stationary perishable goods/* contain most of the code used for the experiments under non-stationary demand, as described in Section 7 of the paper. Each of these folders contains the following files:
- *main.py* can be used to change parameter settings and start the experiments.
- *hyperparams.py* contains code used for hyperparameter optimization.
- *augmentData.py* is used to augment the created training data in line with the given (hyper)parameters.
- *train.py* uses generated data to train a LightGBM model using the specified hyperparameters.
- *evalModel.py* evaluates a trained LightGBM model using simulation runs.
- *utility.py* contains a utility function that can be used to monitor training and tuning progress.

These folders do NOT contain the file *importData.py* and its *importData()* function. They were left out due to confidentiality requirements. The same holds for the data itself.

The folder *Lumpy lost sales/* contains the code used for the experiments under lumpy demand, as described in Appendix B of the paper. It contains the following files:
- *main.py* can be used to change parameter settings and start the experiments.
- *hyperparams.py* contains code used for hyperparameter optimization.
- *createData.py* is used to generate time series training data with a specified length, number of features and (average) demand interval.
- *augmentData.py* is used to augment the created training data in line with the given (hyper)parameters.
- *train.py* uses generated data to train a LightGBM model using the specified hyperparameters.
- *evalModel.py* evaluates a trained LightGBM model using simulation runs.
- *utility.py* contains a utility function that can be used to monitor training and tuning progress.