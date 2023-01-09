# Package imports
import csv

# Local imports
from evalModel import evalModel
from generateData import generateData
from hyperparams import optimize

# Problem parameters
mean = 5 # Expected demand (for the geometric distribution, 1 gets added)
h = 1 # Unit holding cost

# Model parameters
train_size = 1000000 # Number of samples to train on (=1M)
test_size = 10000000 # Number of samples to test on (=10M)
run_size = 10000 # Number of samples per evaluation run (=10K)
warmup = 1000 # Number of samples not to count during evaluation (=1K)
n_jobs = 16 # Number of cores used for data generation, model training and performing predictions
n_duplicates = 3 # Amount of times each parameter configuration gets evaluated (for stability reasons)
n_trials = 42 # Number of trials used by Optuna for hyperparameter tuning
verbose = True # Can be used to turn off most of the messages

# Create dictionary of parameters
params = {"mean": mean, "h": h,
          "train_size": train_size, "n_duplicates": n_duplicates,
          "test_size": test_size, "run_size": run_size, "warmup": warmup,
          "n_jobs": n_jobs, "verbose": verbose}
    
# Create a list of experiments to perform
experiments = []
for distribution in ["Poisson", "geometric"] : # Distribution followed by demand
    for L in [1, 2, 3, 4] : # Leadtime
        for p in [4, 9, 19, 39] : # Unit penalty cost
            params["distribution"] = distribution
            params["L"] = L
            params["p"] = p
            experiments.append(params.copy())

# Find the best model for each of the experiments
tuned_params_list, validation_cost_list = optimize(experiments, n_trials,
                                                   generateData, evalModel)

# Write results to a csv file
with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    for i in range(len(experiments)) :
        writer.writerow([experiments[i], validation_cost_list[i], tuned_params_list[i]])