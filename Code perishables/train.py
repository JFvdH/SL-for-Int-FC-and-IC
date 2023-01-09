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
    m = params['m'] # Time to perish 
    L = params['L'] # Leadtime
    c_t = params['c_t'] # Cost of lost sales
    c_h = params['c_h'] # Unit holding cost
    c_o = params['c_o'] # Per unit order cost
    c_p = params['c_p'] # Perishing cost
    policy = params['policy'] # Issuing policy
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for training
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Custom loss 
    @njit
    def custom_loss_FIFO(_, y_pred):
        
        # Underage component 
        net_I_now = I_afterD + y_pred
        gradient = (net_I_now < 0) * -1 * c_t
        
        # Acquisition component & Cost 'savings' for leftovers
        gradient += (y_pred > 0) * (net_I_now <= 0)  * c_o 
        
        # Holding component 
        gradient += (net_I_now > 0) * c_h
         
        # Overage component
        gradient += (I_expiry + y_pred > 0) * c_p
        
        # Hessian (= set according to L1-norm recommended settings)
        hessian = np.ones(shape=y_pred.shape[0])
        
        # Return the gradient and 'hessian' of the function
        return gradient, hessian
    
    # Custom loss definition
    @njit
    def custom_loss_LIFO(_, y_pred):
    
        # Underage component
        net_I_now = I_afterD + y_pred
        gradient = (net_I_now < 0) * -1 * c_t
        
        # Acquisition component & Cost 'savings' for leftovers (case 1)
        I_now = np.maximum(0, net_I_now)
        P_now = np.minimum(P_line[0,:], I_now)
        net_I_now = I_now - P_now
        gradient += (y_pred > 0) * (net_I_now <= 0) * c_o
        
        # Holding component 
        gradient += (net_I_now > 0) * c_h
        
        # Overage component & Cost 'savings' for leftovers (case 2)
        for i in range(0, m-1) :
            if i > 0 :
                I_now = np.maximum(0, I_now - D[L+i,:])
            P_now = np.minimum(P_line[i,:], I_now)  
            I_now -= P_now
            if (i == 0) or (i == m-2) :
                gradient += (P_now > P_alt[i,:]) * c_p
        net_I_now = I_now - D[-1,:]
        gradient += (net_I_now > 0) * c_p

        # Hessian (= set according to L1-norm recommended settings)
        hessian = np.ones(shape=y_pred.shape[0])
        
        # Return the gradient and 'hessian' of the function
        return gradient, hessian
    
    # Set the appropriate loss function
    if policy == "FIFO" :
        custom_loss = custom_loss_FIFO
    elif policy == "LIFO" :
        custom_loss = custom_loss_LIFO
    else :
        raise ValueError("Policy not recognized")
        
    # Generate relevant columns
    I, D, A, P = genFunc(params, hyperparams)
    X = np.vstack([np.expand_dims(I,0), P, A]).transpose()
    y_dummy = np.zeros(train_size)
    
    # Calculate changes in lead time
    I_leadtime = I.copy()
    P_line = P.copy()
    for i in range(L) :
        # Process arrivals
        if i != 0 :          
            A_period = A[i-1,:].copy()
            A = A[1:,:]
            P_line = np.vstack([P_line, np.expand_dims(A_period, 0)])
            I_leadtime += A_period
        # Process demand
        D_period = D[i,:].copy()
        I_leadtime = np.maximum(0, I_leadtime - D_period)
        # Process expiries
        if policy == "FIFO" :
            for j in range(m) :
                rQty = np.minimum(P_line[j,:], D_period)
                P_line[j,:] -= rQty
                D_period -= rQty
        elif policy == "LIFO" :
            for j in range(m) :
                rQty = np.minimum(P_line[m-1-j,:], D_period)
                P_line[m-1-j,:] -= rQty
                D_period -= rQty      
        I_leadtime -= P_line[0,:]
        P_line = P_line[1:,:]
    I_beforeD = I_leadtime.copy() # Inventory before arrival of the order
    I_afterD = I_leadtime - D[L,:] # Inventory before arrival of the order,
                                   # but with demand already subtracted for computational reasons
    
    # Prepare columns on what would happen in the FIFO case
    if policy == "FIFO" :
        I_expiry = I_beforeD.copy()
        for i in range(L, L+m-1):
            # Process demand
            D_period = D[i,:].copy()
            I_expiry -= D_period 
            # Process expiries
            for j in range(m-i) :
                rQty = np.minimum(P_line[j,:], D_period)
                P_line[j,:] -= rQty
                D_period -= rQty   
            perished = P_line[0,:].copy()
            I_expiry -= perished
            P_line = P_line[1:,:]
        I_expiry -= D[L+m-1,:]
    # Prepare columns on what would happen in the LIFO case
    elif policy == "LIFO" :
        I_expiry = I_beforeD.copy()
        P_line_alt = P_line.copy()
        P_alt = np.zeros(shape=(m-1, train_size))
        for i in range(L, L+m-1):
            # Process demand
            D_period = D[i,:].copy()
            I_expiry -= D_period 
            # Process expiries
            for j in range(m-i-1, -1, -1) :
                rQty = np.minimum(P_line_alt[j,:], D_period)
                P_line_alt[j,:] -= rQty
                D_period -= rQty   
            perished = P_line_alt[0,:].copy()
            P_alt[i-L,:] = perished
            I_expiry -= perished
            P_line_alt = P_line_alt[1:,:]
        I_expiry -= D[L+m-1,:]
    else :
        raise ValueError("Policy not recognized")
            
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