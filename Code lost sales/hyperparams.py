# Package imports
import datetime
from joblib import Parallel, delayed
import optuna

# Local imports
from train import train
from utility import strfdelta

# Function used to evaluate hyperparameter settings
def evalHypers (genFunc, evalFunc, params, hypers) :
    
    # Train the model
    model = train(genFunc, params, hypers)

    # Evaluate the model
    costs = evalFunc(model, params)
    
    # Return the evaluation results
    return costs

# Class to define the Optuna objective
class Objective(object):

    def __init__(self, _params, _genFunc, _evalFunc):
        self.params = _params
        self.genFunc = _genFunc
        self.evalFunc = _evalFunc

    def __call__(self, trial):
        
        # Determining the hyperparameters
        hyperparams = {}
        hyperparams['inv_multiplier'] = trial.suggest_int('inv_multiplier', 2, 64, log=True)
        hyperparams['max_bin'] = trial.suggest_int('max_bin', 32, 512, log=True)
        hyperparams['n_estimators'] = trial.suggest_int("n_estimators", 64, 256, log=True)
        hyperparams['num_leaves'] = trial.suggest_int('num_leaves', 15, 127, log=True)
        hyperparams['min_child_samples'] = trial.suggest_int('min_child_samples', 32, 512, log=True)
        hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 0.0004, 0.08, log=True)
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
        n_duplicates = self.params["n_duplicates"] #
        results = Parallel(n_jobs=n_duplicates)(delayed(evalHypers)(self.genFunc, self.evalFunc,
                                                                    self.params, hyperparams
                                                                    ) for job in range(n_duplicates)) 
        costs = sum(results)/n_duplicates
        if self.params['verbose'] == True :
            delta = datetime.datetime.now() - start_eval
            print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
            print("Est. avg. cost: \t", round(costs, 4)); print("")

        # Return metric of interest
        return costs
    
# Function definition
# Takes set and number of experiments, a generation function and an evaluation function as input
# Outputs corresponding list of best found hyperparameters and associated validation run costs
def optimize(experiments, n_trials, genFunc, evalFunc) :
    
    # Initialize lists of tuned parameters and validation run costs
    tuned_params_list = []
    validation_cost_list = []

    # Loop over all experiments
    n_exps = 1
    for params in experiments:
    
        # Perform the hyperparameter tuning
        start_tuning = datetime.datetime.now()
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        study = optuna.create_study(direction="minimize")
        study.optimize(
            Objective(_params = params, 
                      _genFunc = genFunc, 
                      _evalFunc = evalFunc),
            n_trials = n_trials
        )
        delta = datetime.datetime.now() - start_tuning
        print("#### Finished tuning experiment", n_exps, "####")
        print("Time to tune: \t\t", strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print('Nr. of trials:\t\t', len(study.trials))
        print('Best trial:', study.best_trial.params); print("")
        tuned_params_list.append(study.best_params.copy())
        n_exps += 1
        
        # Evaluate the tuned model
        start_eval = datetime.datetime.now()
        print("Starting evaluation of the best found hyperparameters...")
        n_duplicates = params["n_duplicates"] #
        results = Parallel(n_jobs=n_duplicates)(delayed(evalHypers)(genFunc, evalFunc,
                                                                    params, study.best_params
                                                                    ) for job in range(n_duplicates)) 
        costs = sum(results)/n_duplicates
        validation_cost_list.append(costs.copy())
        delta = datetime.datetime.now() - start_eval
        print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print("Est. avg. cost: \t", round(costs, 4)); print("")
    
    # Return the lists of tuned parameters and validation run costs
    return tuned_params_list, validation_cost_list