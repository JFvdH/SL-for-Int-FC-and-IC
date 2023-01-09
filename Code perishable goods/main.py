# Package imports
import csv

# Local imports
from evalModel import evalModel
from generateData import generateData
from hyperparams import optimize

# Initialize problem parameters
c_t = 5 # Cost of lost sales
c_h = 1 # Unit holding cost
c_o = 3 # Per unit order cost
mu = 4 # Mean
sigma = 2 # Standard deviation

# Initialize model parameters
train_size = 1000000 # Number of samples to train on (=1M)
test_size = 10000000 # Number of samples to test on (=10M)
run_size = 10000 # Number of samples per evaluation run (=10K)
warmup = 1000 # Number of samples not to count during evaluation (=1K)
n_jobs = 16 # Number of cores used for data generation, model training and performing predictions
n_duplicates = 3 # Amount of times each parameter configuration gets evaluated (for stability reasons)
n_trials = 42 # Number of trials used by Optuna for hyperparameter tuning
verbose = True # Can be used to turn off most of the messages

# Use parameters to derive key variables
a = mu**2/sigma**2 # Shape parameter of the gammma distribution
scale = sigma**2/mu # Scale parameter of the gamma distribution
params = {"c_t": c_t, "c_h": c_h, "c_o": c_o,
          "mu": mu, "sigma": sigma,
          "train_size": train_size, "n_duplicates": n_duplicates,
          "test_size": test_size, "run_size": run_size, "warmup": warmup,
          "n_jobs": n_jobs, "verbose": verbose}

# Create a list of experiments to perform
experiments = []
for m in [2, 3, 4, 5] : # Time to perish
    for L in [1, 2] : # Leadtime
        for c_p in [7, 10] : # Cost of perishing
            for policy in ["LIFO"]: #["FIFO", "LIFO"] : # Issuing policy
                params["m"] = m
                params["L"] = L
                params["c_p"] = c_p
                params["policy"] = policy
                experiments.append(params.copy())

# Find the best model for each of the experiments
tuned_params_list, validation_cost_list = optimize(experiments, n_trials,
                                                   generateData, evalModel)

# Write results to a csv file
with open('results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    for i in range(len(experiments)) :
        writer.writerow([experiments[i], validation_cost_list[i], tuned_params_list[i]])

