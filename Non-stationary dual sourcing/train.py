# Package imports
import contextlib
import datetime
import lightgbm as lgbm
from numba import njit
import numpy as np
import pandas as pd

# Package imports
import copy
from warnings import simplefilter

# Local imports
from utility import strfdelta

# Definition of the training function
# Takes parameters as input and returns a trained model
def train(data, params, hyperparams, verbose = False) :

    # Suppress warnings from LightGBM
    simplefilter("ignore", category=Warning)    

    # Initialize problem parameters
    h = params['h'] # Unit holding cost
    b = params['b'] # Unit backorder cost
    c_r = params['c_r'] # Regular unit order cost
    c_e = params['c_e'] # Expediting unit order cost
    l_r = int(data.iloc[0]['l_r']) # Regular leadtime
    L_max = params["L_max"] # Maximum leadtime
    
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
        X_exp["I_exp"] = I_reg_leadtimediff + np.maximum(0, y_reg) 
        y_exp_raw = np.round(model_exp.predict(X_exp, n_jobs=n_jobs))
    
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
        
    # Generate relevant columns
    I_exp = data.loc[:,'I_exp'].to_numpy()
    I_reg = data.loc[:,'I_reg'].to_numpy()
    D_reg = data.loc[:,['D_'+str(i) for i in range(1, L_max+2)]].to_numpy()
    A_reg = data.loc[:,['A_'+str(i) for i in range(1, L_max)]].to_numpy()
    D_last_exp = data['D_'+str(1)].to_numpy().astype(int)
    D_last_reg = np.zeros(train_size)
    for i in range(1, l_r+2) :
        D_last_reg += data['D_'+str(i)].mul(data['l_r']==i-1)
    D_last_reg = D_last_reg.to_numpy().astype(int)
    X = data.drop(['demand', 'date', 'store_closed', 
                   'l_r'] + ['D_'+str(i) for i in range(1, L_max+2)] + list(data.filter(regex='__')), 
                  axis=1)
    y_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_exp_leadtime = I_exp
    I_exp_leadtime -= D_last_exp
    
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
        print(""); print("Starting model training...")
    start_train = datetime.datetime.now()
    with contextlib.redirect_stdout(None): # Suppress output during training
        model_exp.fit(X.drop(['I_reg'] + ['A_'+str(i) for i in range(1, L_max)], axis=1), y_dummy)
    if verbose == True :
        delta = datetime.datetime.now() - start_train
        print("Finished model training after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s')); print("")
        
    # Recall the column names
    variableColumns = [] # Note: Column names are masked in this publicized code due to confidentiality
    fixedColumns = [] # Note: Column names are masked in this publicized code due to confidentiality
    fixedData = data[fixedColumns]
        
    # Calculate changes in lead time
    I_reg_leadtimediff = I_reg
    for i in range(l_r) :
        # Process arrivals
        if i != 0 :
            I_reg_leadtimediff += A_reg[:,i-1]
        # Derive and process expedited orders
        inventoryData = pd.DataFrame(np.expand_dims(I_reg_leadtimediff,1), columns=['I_exp'])  
        varCols = [str(i)+"__"+column for column in copy.deepcopy(variableColumns)]
        variableData = data[varCols].copy()
        variableData.rename(columns=lambda x: x[3:], inplace=True)
        X_exp_local = pd.concat([inventoryData, fixedData, variableData], axis=1)
        y_exp_local = np.maximum(0, np.round(model_exp.predict(X_exp_local, n_jobs=n_jobs))).astype(int)
        I_reg_leadtimediff += y_exp_local
        if i == 0 :
            X["I_reg"] += y_exp_local
        # Process demand 
        I_reg_leadtimediff = np.maximum(0, I_reg_leadtimediff - D_reg[:,i])
    I_reg_leadtime = I_reg_leadtimediff - D_last_reg # Inventory before arrival of the order, but with
                                                      # demand already subtracted for computational reasons
    
    # Prepare DataFrame for last-moment expediting decisions
    inventoryData = pd.DataFrame(np.expand_dims(np.zeros(len(I_reg_leadtime)),1).astype(int), columns=['I_exp'])    
    varCols = [str(l_r)+"__"+column for column in copy.deepcopy(variableColumns)]
    variableData = data[varCols].copy()
    variableData.rename(columns=lambda x: x[3:], inplace=True)
    X_exp = pd.concat([inventoryData, fixedData, variableData], axis=1)
    
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
        print(""); print("Starting model training...")
    start_train = datetime.datetime.now()
    with contextlib.redirect_stdout(None): # Suppress output during training
        model_reg.fit(X.drop(['I_exp'], axis=1), y_dummy)
    if verbose == True :
        delta = datetime.datetime.now() - start_train
        print("Finished model training after " + strfdelta(delta, '{H:02}h:{M:02}m:{S:02}s')); print("")
        
    # Return the trained model
    return model_exp, model_reg