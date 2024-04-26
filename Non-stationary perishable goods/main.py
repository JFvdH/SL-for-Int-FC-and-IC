# Package imports
import csv

# Local imports
from hyperparams import optimize

# Problem parameters
c_h = 1 # Unit holding cost
c_p = 8 # Unit perishing cost
c_o = 3 # Per unit order cost
L_max = 2 # Maximum leadtime

# Model parameters
train_size = 1000000 # Number of samples to train on (=1M)
months_val = 12 # Amount of months to use for validation
n_jobs = 4 # Number of cores used for data generation, model training and performing predictions
n_duplicates = 12 # Amount of times each parameter configuration gets evaluated (for stability reasons)
n_trials = 42 # Number of trials used by Optuna for hyperparameter tuning
D_index = 0 # Index of the column containing demand
verbose = True # Can be used to turn off most of the messages

# Create a dictionary of parameters
params = {"c_h": c_h, "c_o": c_o, "c_p": c_p,
          "train_size": train_size, 'months_val': months_val , 
          'D_index': D_index, 'L_max': L_max,
          "n_jobs": n_jobs, "n_duplicates": n_duplicates,
          "verbose": verbose}

# Create a set of experiments
experiments = []
for c_t in [5, 8, 12, 17] : # Unit lost sales cost
        for policy in ["FIFO", "LIFO"] : # Issuing policy
            for subgroup in [40, 210, 328, 125] : # Product category
                params['c_t'] = c_t
                params['policy'] = policy
                params['subgroup'] = subgroup
                experiments.append(params.copy())

# Find the best model for each of the experiments
tuned_params_list, validation_cost_list = optimize(experiments, n_trials)

# Write results to a csv file
with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    for i in range(len(experiments)) :
        writer.writerow([experiments[i], validation_cost_list[i], tuned_params_list[i]])