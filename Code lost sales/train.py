# Package imports
import datetime
import lightgbm as lgbm
from numba import njit
import numpy as np

# Local imports
from utility import strfdelta

# Definition of the training function
# Takes parameters as input and returns a trained model
def train(genFunc, params, hyperparams, verbose = False) :

    # Initialize problem parameters
    h = params['h'] # Unit holdig cost
    p = params['p'] # Unit penalty cost
    L = params['L'] # Leadtime
    
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
    I, D, A = genFunc(params, hyperparams)
    X = np.hstack([np.expand_dims(I,1), A.transpose()])
    y_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_leadtime = I.copy()
    for i in range(L) :
        if i != 0 :
            I_leadtime += A[i-1,:]
        I_leadtime = np.maximum(0, I_leadtime - D[i,:])
    I_leadtime -= D[-1,:] # Inventory before arrival of the order,
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