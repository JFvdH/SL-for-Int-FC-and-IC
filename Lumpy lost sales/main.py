# Package imports
import csv
import warnings

# Local imports
from hyperparams import optimize

# Problem parameters
h = 1 # Unit holding cost
L = 0 # Leadtime 

# Model parameters
train_size = 1000000 # Number of samples to train on (=1M)
n_eval = 1 # Number of models that are trained and evaluated per setting
n_jobs = 12 # Number of cores used for data generation, model training and performing predictions
n_trials = 42*2 # Number of trials used by Optuna for hyperparameter tuning
verbose = True # Can be used to turn off most of the messages

# Deriving dictionary of parameters
params = {'h': h, 'L': L, 'n_eval': n_eval,
          'train_size': train_size, 
          'n_jobs': n_jobs, 'verbose': verbose}

# Create a set of experiments
experiments = []
for p in [4, 9 ,19, 39] :
    params['p'] = p
    for freq in [1, 2, 3, 4] :
        params['freq'] = freq
        experiments.append(params.copy())

# Find the best model for each of the experiments
warnings.simplefilter(action='ignore', category=FutureWarning)
tuned_params_list, validation_cost_list = optimize(experiments, n_trials)

# Write results to a csv file
with open('results.csv', 'w', newline='') as results_file :
    writer = csv.writer(results_file)
    for i in range(len(experiments)) :
        writer.writerow([experiments[i], validation_cost_list[i], tuned_params_list[i]])