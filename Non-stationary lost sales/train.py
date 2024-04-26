# Package imports
import datetime
import lightgbm as lgbm
from numba import njit
import numpy as np

# Package imports
import copy

# Local imports
from utility import strfdelta

# Definition of the training function
# Takes parameters as input and returns a trained model
def train(data, params, hyperparams, verbose = False) :

    # Initialize problem parameters
    h = params['h'] # Unit holdig cost, incl. salvage cost
    p = params['p'] # Unit penalty cost
    L = int(data["L"].iloc[0]) # Leadtime of the product group
    L_max = params['L_max'] # Leadtime of the product group with the longest leadtime
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for training
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Custom loss 
    @njit
    def custom_loss(_, y):
                
        # Holding component
        gradient = (I_leadtime + y > 0) * h 
        
        # Underage component
        gradient += (I_leadtime + y < 0) * -1 * p
        
        # Hessian (= set according to L1-norm recommended settings)
        hessian = np.ones(shape=y.shape[0])
        
        # Return the gradient and 'hessian' of the function
        return gradient, hessian
        
    # Generate relevant columns
    I = copy.deepcopy(data.loc[:,'I'].to_numpy())
    D = copy.deepcopy(data.loc[:,['D_'+str(i) for i in range(1, L_max+2)]].to_numpy())
    A = copy.deepcopy(data.loc[:,['A_'+str(i) for i in range(1, L_max)]].to_numpy())
    D_last = np.zeros(train_size)
    for i in range(1, L+2) :
        D_last += data['D_'+str(i)].mul(data['L']==i-1)
    D_last = D_last.to_numpy()
    X = data.drop(['demand', 'date', 'store_closed', 
                   'L'] + ['D_'+str(i) for i in range(1, L_max+2)], axis=1)
    y_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_leadtime = I.copy()
    for i in range(L) :
        if i != 0 :
            I_leadtime += A[:,i-1]
        I_leadtime = np.maximum(0, I_leadtime - D[:,i])
    I_leadtime -= D_last # Inventory before arrival of the order,
                         # but with demand already subtracted for computational reasons
    
    # Model definition
    model = lgbm.LGBMRegressor(objective=custom_loss, verbose = -1, n_jobs = n_jobs,
                               subsample = 0.5, subsample_freq = 1,
                                max_bin = hyperparams["max_bin"], 
                                n_estimators = hyperparams["n_estimators"],
                                num_leaves = hyperparams["num_leaves"], 
                                min_child_samples = hyperparams["min_child_samples"],
                                learning_rate = hyperparams["learning_rate"],
                                reg_alpha = hyperparams["reg_alpha"], 
                                reg_lambda = hyperparams["reg_lambda"])
    
    # Model training and update print statements
    if verbose == True :
        print(""); print("Starting model training...")
    start_train = datetime.datetime.now()
    model.fit(X, y_dummy)
    if verbose == True :
        delta = datetime.datetime.now() - start_train
        print("Finished model training after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s')); print("")
    
    # Return the trained model
    return model