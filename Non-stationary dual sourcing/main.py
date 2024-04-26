# Package imports
import csv

# Local imports
from hyperparams import optimize

# Problem parameters
h = 1 # Unit holding cost
b = 79 # Unit backorder cost
c_r = 0 # Regular unit cost
L_max = 4 # Longest leadtime

# Model parameters
train_size = 1000000 # Number of samples to train on (=1M)
months_val = 12 # Amount of months to use for validation
n_jobs = 4 # Number of cores used for data generation, model training and performing predictions
n_duplicates = 12 # Amount of times each parameter configuration gets evaluated (for stability reasons)
n_trials = 42 # Number of trials used by Optuna for hyperparameter tuning
D_index = 0 # Index of the column containing demand
verbose = True # Can be used to turn off most of the messages

# Deriving dictionary of parameters
params = {"h": h, "b": b, "c_r": c_r,
          'D_index': D_index, 'L_max': L_max,
          'train_size': train_size, 'months_val': months_val ,
          'n_jobs': n_jobs, 'n_duplicates': n_duplicates,
          'verbose': verbose}

# Create a set of experiments
experiments = []
for c_e in [4, 9 ,19, 39] :
    for subgroup in [82, 248, 109, 219] :
        params['c_e'] = c_e
        params['subgroup'] = subgroup
        experiments.append(params.copy())

# Find the best model for each of the experiments
tuned_params_list, validation_cost_list = optimize(experiments, n_trials)

# Write results to a csv file
with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    for i in range(len(experiments)) :
        writer.writerow([experiments[i], validation_cost_list[i], tuned_params_list[i]])