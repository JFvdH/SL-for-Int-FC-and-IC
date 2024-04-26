# Package imports
import copy
import datetime
from joblib import Parallel, delayed
import numpy as np
import optuna

# Local imports
from augmentData import augmentData
from evalModel import evalModel
from importData import importData
from train import train
from utility import subtractMonths, strfdelta

# Function used to evaluate hyperparameter settings
def evalHypers(input_data, month, params, hypers, verbose=False) :
    
    # Make a deep copy of the data to prevent pointer issues
    data = copy.deepcopy(input_data)

    # Determine cut-off dates
    last_date = str(data['date'].max())[:10]
    train_cutoff_date = subtractMonths(last_date, month)
    val_cutoff_start = subtractMonths(last_date, month+3)
    val_cutoff_end = subtractMonths(last_date, month-1)
    
    # Obtain the desired datasets
    train_set = data[data['date'] <= train_cutoff_date] 
    val_set = data[(data['date'] > val_cutoff_start) & (data['date'] <= val_cutoff_end)]
    
    # Train and evaluate multiple models with the same hyperparameters (for stability reasons)
    aTrain_set = augmentData(train_set, params, hypers)
    model = train(aTrain_set, params, hypers)
    costs = evalModel(val_set, train_cutoff_date, model, params, verbose)
        
    # Return the results
    return costs

# Class to define the Optuna objective
class Objective(object):

    def __init__(self, _val_set, _params):
        self.val_set = _val_set
        self.params = _params

    def __call__(self, trial):
        
        # Determining the hyperparameters
        hyperparams = {}
        hyperparams['inv_multiplier'] = trial.suggest_int('inv_multiplier', 2, 64, log=True)
        hyperparams['max_bin'] = trial.suggest_int('max_bin', 32, 512, log=True)
        hyperparams['n_estimators'] = trial.suggest_int("n_estimators", 128, 512, log=True)
        hyperparams['num_leaves'] = trial.suggest_int('num_leaves', 15, 127, log=True)
        hyperparams['min_child_samples'] = trial.suggest_int('min_child_samples', 32, 512, log=True)
        hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 0.0008, 0.16, log=True)
        hyperparams['reg_alpha'] = trial.suggest_float('reg_alpha', 0.000002, 4, log=True)
        hyperparams['reg_lambda'] = trial.suggest_float('reg_lambda', 0.000002, 4, log=True)
        
        # Evaluate the model
        if self.params['verbose'] == True :
            start_eval = datetime.datetime.now()
            print("Starting evaluation with the following hyperparameters...")
            print("Inv. multiplier: \t", hyperparams["inv_multiplier"])   
            print("Nr. of estimators: \t", hyperparams["n_estimators"])   
            print("Nr. of leaves: \t\t", hyperparams["num_leaves"])
            print("Learning rate: \t\t", round(hyperparams["learning_rate"],4)) 
        n_duplicates = self.params["n_duplicates"] 
        months_val = self.params["months_val"]
        results = Parallel(n_jobs=n_duplicates)(delayed(evalHypers)(self.val_set, i, self.params, hyperparams
                                                                    ) for i in range(1, months_val+1)) 
        costs = sum(results)/months_val
        if self.params['verbose'] == True :
            delta = datetime.datetime.now() - start_eval
            print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
            print("Avg. std. cost: \t", round(costs, 2)); print("")

        # Return metric of interest
        return costs
    
# Function definition
# Takes set and number of experiments, a generation function and an evaluation function as input
# Outputs corresponding list of best found hyperparameters and associated validation run costs
def optimize(experiments, n_trials) :
    
    # Initialize lists of tuned parameters and validation run costs
    tuned_params_list = []
    evaluation_cost_list = []

    # Loop over all experiments
    n_exps = 1
    for params in experiments:
    
        # Data import
        subgroup = params['subgroup']
        val_set, test_set = importData(subgroup)
    
        # Perform the hyperparameter tuning
        start_tuning = datetime.datetime.now()
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        study = optuna.create_study(direction="minimize")
        study.optimize(
            Objective(_val_set = val_set,
                      _params = params),
            n_trials = n_trials
        )
        delta = datetime.datetime.now() - start_tuning
        print("#### Finished hyperparameter tuning experiment", n_exps, 
                         "#### (subgroup "+str(subgroup)+", "+str(params['policy'])+" policy, c_t="+str(params['c_t'])+")")
        print("Time to tune: \t\t", strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print('Nr. of trials:\t\t', len(study.trials))
        print('Best trial:', study.best_trial.params); print("")
        tuned_params_list.append(study.best_params.copy())
        n_exps += 1
        
        # Evaluate the tuned model
        start_eval = datetime.datetime.now()
        print("Starting evaluation of the best found hyperparameters...")
        n_duplicates = params["n_duplicates"]
        months_val = params["months_val"]
        results = Parallel(n_jobs=n_duplicates)(delayed(evalHypers)(test_set, i, params, study.best_params,
                                                                    ) for i in range(1, months_val+1)) 
        costs = sum(results)/months_val
        error_margin = np.std(results, ddof=1)
        evaluation_cost_list.append(costs.copy())
        delta = datetime.datetime.now() - start_eval
        print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print("Avg. est. cost: \t", round(costs, 2), "+-", round(error_margin, 2))
        print("Run costs:", np.array(results).round(2)); print("")
    
    # Return the lists of tuned parameters and validation run costs
    return tuned_params_list, evaluation_cost_list