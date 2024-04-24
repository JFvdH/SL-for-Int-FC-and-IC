# Package imports
import datetime
import optuna

# Local imports
from augmentData import augmentData
from evalModel import evalModel
from createData import createData
from train import train
from utility import strfdelta

# Class to define the Optuna objective
class Objective(object):

    def __init__(self, _train_set, _val_set, _params):
        self.train_set = _train_set
        self.val_set = _val_set
        self.params = _params

    def __call__(self, trial):
        
        # Determining the hyperparameters
        hyperparams = {}
        hyperparams['inv_multiplier'] = trial.suggest_float('inv_multiplier', 0.2, 4, log=True)
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
        costs = 0
        for i in range(self.params['n_eval']) :
            aTrain_set = augmentData(self.train_set, self.params, hyperparams)
            model = train(aTrain_set, self.params, hyperparams)
            costs += evalModel(self.val_set, model, self.params)
        costs = costs / self.params['n_eval']
        if self.params['verbose'] == True :
            delta = datetime.datetime.now() - start_eval
            print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
            print("Avg. std. cost: \t", round(costs, 2)); print("")

        # Return metric of interest
        return costs
    
# Function definition
# Takes set of experiments and number of Optuna trials as input
# Outputs corresponding list of best found hyperparameters and associated validation run costs
def optimize(experiments, n_trials) :
    
    # Initialize lists of tuned parameters and validation run costs
    tuned_params_list = []
    evaluation_cost_list = []

    # Loop over all experiments
    n_exps = 1
    for params in experiments:
    
        # Data import
        train_set, val_set, test_set = createData(frequency = params['freq'])
    
        # Perform the hyperparameter tuning
        start_tuning = datetime.datetime.now()
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        study = optuna.create_study(direction="minimize")
        study.optimize(
            Objective(_train_set = train_set,
                      _val_set = val_set,
                      _params = params),
            n_trials = n_trials
        )
        delta = datetime.datetime.now() - start_tuning
        print("#### Finished hyperparameter tuning experiment", n_exps, 
                         "#### (p="+str(params['p'])+", freq.="+str(params['freq'])+")")
        print("Time to tune: \t\t", strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print('Nr. of trials:\t\t', len(study.trials))
        print('Best trial:', study.best_trial.params); print("")
        tuned_params_list.append(study.best_params.copy())
        n_exps += 1
        
        # Evaluate the tuned model
        start_eval = datetime.datetime.now()
        print("Starting evaluation of the best found hyperparameters...")
        costs = 0
        for i in range(params['n_eval']) :
            aTrain_set = augmentData(train_set, params, tuned_params_list[-1])
            model = train(aTrain_set, params, tuned_params_list[-1])
            costs += evalModel(test_set, model, params)
        costs = costs / params['n_eval']
        evaluation_cost_list.append(costs.copy())
        delta = datetime.datetime.now() - start_eval
        print("Finished evaluation after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s'))
        print("Avg. est. cost: \t", round(costs, 2))
        print("Run costs:", costs); print("")
    
    # Return the lists of tuned parameters and validation run costs
    return tuned_params_list, evaluation_cost_list