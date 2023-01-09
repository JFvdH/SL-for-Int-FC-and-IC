# Package imports
import contextlib
import datetime
import lightgbm as lgbm
from numba import njit
import numpy as np
from warnings import simplefilter
    
# Local imports
from utility import strfdelta

# Definition of the training function
# Takes parameters as input and returns a trained model
def train(genFunc, params, hyperparams, verbose = False) :

    # Suppress warnings from LightGBM
    simplefilter("ignore", category=Warning)    

    # Initialize problem parameters
    h = params['h'] # Unit holding cost
    b = params['b'] # Unit backorder cost
    c_r = params['c_r'] # Regular unit order cost
    c_e = params['c_e'] # Expediting unit order cost
    l_r = params['l_r'] # Regular leadtime
    l_e = params['l_e'] # Expediting # TODO: Make the implementation work for l_e > 0
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for training
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Custom loss definition for expediting model
    @njit
    def custom_loss_exp(_, y_exp):
        
        # Underage component
        gradient = (I_exp_leadtime + y_exp < 0) * -1 * b
        
        # Holding component
        gradient += (I_exp_leadtime + y_exp > 0) * h
        
        # Acquisition component
        gradient += (c_e - c_r) 
        
        # Hessian (= set according to L1-norm recommended settings)
        hessian = np.ones(shape=y_exp.shape[0])
        
        # Return the gradient and 'hessian' of the function
        return gradient, hessian

    # Custom loss definition for regular model
    def custom_loss_reg(_, y_reg):
    
        # Retrieve expediting model actions
        sX_exp =  np.expand_dims(I_reg_leadtimediff + np.maximum(0, y_reg), 1)
        y_exp_raw = np.round(model_exp.predict(sX_exp, n_jobs=n_jobs))
    
        # Calculate the gradient and provide the hessian
        return custom_loss_reg_speedup(y_reg, y_exp_raw)
    
    @njit
    def custom_loss_reg_speedup(y_reg, y_exp_raw) :
        
        # Smoothen y_exp
        y_exp = np.maximum(0, y_exp_raw)
        
        # Underage and holding components
        I_end = I_reg_leadtime + y_reg + y_exp
        gradient = (I_end < 0) * (y_exp == 0) * -1 * b
        gradient += (I_end > 0) * (y_exp == 0) * h # Only pay holding if it wouldn't be there otherwise
    
        # Cost savings for regular orders
        gradient += (y_exp > 0) * -1 * (c_e - c_r) # Cost savings to prevent y_exp
        
        # Hessian (= set according to L1-norm recommended settings)
        hessian = np.ones(shape=y_reg.shape[0])
        
        # Return the gradient and 'hessian' of the function
        return gradient, hessian

    # Generate relevant columns for the expediting model
    altParams = params.copy()
    altParams['reg'] = False
    I_exp, D_exp, A_exp = genFunc(altParams, hyperparams)
    X_exp = np.expand_dims(I_exp, 1)
    y_exp_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_exp_leadtime = I_exp.copy()
    I_exp_leadtime -= D_exp[l_e:l_e+train_size]

    # Expediting model definition
    model_exp = lgbm.LGBMRegressor(objective=custom_loss_exp, verbose = -1, n_jobs = n_jobs,
                                   subsample = 0.5, subsample_freq = 1,
                                   max_bin = hyperparams["max_bin_1"], 
                                   n_estimators = hyperparams["n_estimators_1"],
                                   num_leaves = hyperparams["num_leaves_1"], 
                                   min_child_samples = hyperparams["min_child_samples_1"],
                                   learning_rate = hyperparams["learning_rate_1"],
                                   reg_alpha = hyperparams["reg_alpha_1"], 
                                   reg_lambda = hyperparams["reg_lambda_1"])
    
    # Model training and update print statements
    if verbose == True :
        print(""); print("Starting expediting model training...")
    start_train = datetime.datetime.now()
    with contextlib.redirect_stdout(None): # Suppress output during training
        model_exp.fit(X_exp, y_exp_dummy)
    if verbose == True :
        delta = datetime.datetime.now() - start_train
        print("Finished expediting model training after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s')); print("")
        
    # Generate relevant columns for the regular model
    altParams = params.copy()
    altParams['reg'] = True
    I_reg, D_reg, A_reg = genFunc(altParams, hyperparams)
    X_reg = np.vstack([I_reg] + [A_reg[i:train_size+i] for i in range(l_r-1)]).transpose()
    y_reg_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_temp = I_reg[:train_size].copy()[:train_size]
    for i in range(l_r) :
        I_temp += np.maximum(0, np.round(model_exp.predict(np.expand_dims(I_temp, 1)))).astype(int)
        I_temp -= D_reg[i:train_size+i]
        if i < l_r - 1:
            I_temp += A_reg[i:train_size+i] 
    I_reg_leadtimediff = I_temp.copy()[:train_size]
    I_reg_leadtime = I_temp.copy() - D_reg[l_r:l_r+train_size]
        
    # Regular model definition
    model_reg = lgbm.LGBMRegressor(objective=custom_loss_reg, verbose = -1, n_jobs = n_jobs, 
                                   subsample = 0.5, subsample_freq = 1,
                                   max_bin = hyperparams["max_bin_2"], 
                                   n_estimators = hyperparams["n_estimators_2"],
                                   num_leaves = hyperparams["num_leaves_2"], 
                                   min_child_samples = hyperparams["min_child_samples_2"],
                                   learning_rate = hyperparams["learning_rate_2"],
                                   reg_alpha = hyperparams["reg_alpha_2"], 
                                   reg_lambda = hyperparams["reg_lambda_2"])
    
    # Model training and update print statements
    if verbose == True :
        print(""); print("Starting regular model training...")
    start_train = datetime.datetime.now()
    with contextlib.redirect_stdout(None): # Suppress output during training
        model_reg.fit(X_reg, y_reg_dummy)
    if verbose == True :
        delta = datetime.datetime.now() - start_train
        print("Finished regular model training after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s')); print("")
        
    # Return the trained model
    return model_exp, model_reg