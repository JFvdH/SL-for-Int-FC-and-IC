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
    model_exp, model_reg = train(aTrain_set, params, hypers)
    costs = evalModel(val_set, train_cutoff_date, model_exp, model_reg, params, verbose)
        
    # Return the results
    return costs

# Class to define the Optuna objective
class Objective(object):

    def __init__(self, _val_set, _params):
        self.val_set = _val_set
        self.params = _params

    def __call__(self, trial):
        
        # Determining the expediting model hyperparameters (w/ tighter bounds to restrict running time)
        hyperparams = {}
        hyperparams['inv_multiplier_1'] = trial.suggest_int('inv_multiplier_1', 2, 64, log=True)
        hyperparams['max_bin_1'] = trial.suggest_int('max_bin_1', 16, 256, log=True)
        hyperparams['n_estimators_1'] = trial.suggest_int("n_estimators_1", 32, 128, log=True)
        hyperparams['num_leaves_1'] = trial.suggest_int('num_leaves_1', 7, 63, log=True)
        hyperparams['min_child_samples_1'] = trial.suggest_int('min_child_samples_1', 32, 512, log=True)
        hyperparams['learning_rate_1'] = trial.suggest_float('learning_rate_1', 0.0008, 0.16, log=True)
        hyperparams['reg_alpha_1'] = trial.suggest_float('reg_alpha_1', 0.000002, 4, log=True)
        hyperparams['reg_lambda_1'] = trial.suggest_float('reg_lambda_1', 0.000002, 4, log=True)
        
        # Determining the regular model hyperparameters
        hyperparams['inv_multiplier_2'] = trial.suggest_int('inv_multiplier_2', 2, 64, log=True)
        hyperparams['max_bin_2'] = trial.suggest_int('max_bin_2', 32, 512, log=True)
        hyperparams['n_estimators_2'] = trial.suggest_int("n_estimators_2", 64, 256, log=True)
        hyperparams['num_leaves_2'] = trial.suggest_int('num_leaves_2', 15, 127, log=True)
        hyperparams['min_child_samples_2'] = trial.suggest_int('min_child_samples_2', 32, 512, log=True)
        hyperparams['learning_rate_2'] = trial.suggest_float('learning_rate_2', 0.0016, 0.32, log=True)
        hyperparams['reg_alpha_2'] = trial.suggest_float('reg_alpha_2', 0.000002, 4, log=True)
        hyperparams['reg_lambda_2'] = trial.suggest_float('reg_lambda_2', 0.000002, 4, log=True)
        
        # Evaluate the model
        if self.params['verbose'] == True :
            start_eval = datetime.datetime.now()
            print("Starting evaluation with the following regular model hyperparameters...")
            print("Exp. inv. multiplier: \t", hyperparams["inv_multiplier_1"])   
            print("Reg. inv. multiplier: \t", hyperparams["inv_multiplier_2"])   
            print("Exp. nr. of estimators:\t", hyperparams["n_estimators_1"])   
            print("Reg. nr. of estimators:\t", hyperparams["n_estimators_2"])   
            print("Exp. learning rate:\t\t", round(hyperparams["learning_rate_1"], 4)) 
            print("Reg. learning rate:\t\t", round(hyperparams["learning_rate_2"], 4)) 
        n_duplicates = self.params["n_duplicates"] 
        months_val = self.params["months_val"]
        results = Parallel(n_jobs=n_duplicates)(delayed(evalHypers)(self.val_set, i, self.params, hyperparams
                                                                    ) for i in range(1, months_val+1)) 
        costs = sum(results)/months_val
        if self.params['verbose'] == True :
            delta = datetime.datetime.now() - start_eval
            print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
            print("Avg. std. cost: \t\t", round(costs, 2)); print("")

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
                         "#### (subgroup "+str(subgroup)+", c_e="+str(params['c_e'])+")")
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