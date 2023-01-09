# Package imports
import numpy as np

# Evaluate function
# Takes a model as input and returns an estimate of the average long run costs
def evalModel(model, params, verbose = False) :
    
    # Initialize problem parameters
    m = params['m'] # Time to perish 
    L = params['L'] # Lead time 
    c_t = params['c_t'] # Cost of lost sales
    c_h = params['c_h'] # Unit holding cost
    c_o = params['c_o'] # Per unit order cost
    c_p = params['c_p'] # Perishing cost
    mu = params['mu'] # Mean
    sigma = params['sigma'] # Standard deviation
    
    # Initialize model parameters
    policy = params['policy'] # Issuing policy
    test_size = params['test_size'] # Amount of sample data used for training
    run_size = params['run_size'] # Amount of samples generated per run
    warmup = params['warmup'] # Warm-up period of the simulation (= not counted)
    n_jobs = params['n_jobs'] # Amount of cores used by LightGBM

    # Use parameters to derive key variables
    a = mu**2/sigma**2 # Shape parameter of the gammma distribution
    scale = sigma**2/mu # Scale parameter of the gamma distribution
    n_runs = round(test_size/run_size) # Number of runs to execute
    
    # Initialize the system
    demand = np.random.gamma(a, scale, size=(n_runs, run_size+warmup)).round().astype(int)
    I = (np.ones(n_runs)*mu).astype(int)
    perishline = (np.zeros(shape=(n_runs, m))).astype(int)
    pipeline = (np.ones(shape=(n_runs, L))*mu).astype(int)
    
    # Initialize metrics
    total_lost = np.zeros(n_runs)
    total_hold = np.zeros(n_runs)
    total_perish = np.zeros(n_runs)
    total_order = np.zeros(n_runs)

    # Go over all inventory cycles in the different runs
    for i in range(run_size+warmup) :
        
        # Do not count the warmup period
        if i == warmup - 1 :
            total_lost = np.zeros(n_runs)
            total_hold = np.zeros(n_runs)
            total_perish = np.zeros(n_runs)
            total_order = np.zeros(n_runs)
        
        # Calculate inventory after demand, but before perishing and new arrivals
        D = demand[:,i]
        total_lost += np.maximum(0, D-I)
        I = np.maximum(0, I-D)
        
        # Update the perishing pipeline w.r.t. demand
        if policy == "FIFO" :
            for j in range(m) :
                rQty = np.minimum(perishline[:,j], D)
                perishline[:,j] -= rQty
                D -= rQty
        elif policy == "LIFO" :
            for j in range(m) :
                rQty = np.minimum(perishline[:,m-1-j], D)
                perishline[:,m-1-j] -= rQty
                D -= rQty        
        else :
            raise ValueError("Policy not recognized")
        
        # Update the perishing pipeline w.r.t. new arrivals
        perished = perishline[:,0].copy()
        perishline = perishline[:,1:]
        I -= perished
        total_hold += I
        total_perish += perished
        arrival = pipeline[:,0].copy()
        pipeline = pipeline[:,1:]
        perishline = np.hstack([perishline, np.expand_dims(arrival, 1)])
        
        # Process the new arrival and update the arrival pipeline
        I += arrival
        
        # Prepare model input
        model_input = np.hstack([np.expand_dims(I, 1), perishline, pipeline])
        
        # Make predictions and process the results
        prediction = model.predict(model_input, n_jobs=n_jobs)
        prediction = np.floor(np.maximum(0, prediction)).astype(int)
        total_order += prediction
        pipeline = np.hstack([pipeline, np.expand_dims(prediction, 1)])
   
    # Determine the average cost per time unit    
    cost_lostsales = c_t * np.mean(total_lost) / run_size
    cost_holding = c_h * np.mean(total_hold) / run_size
    cost_perish = c_p * np.mean(total_perish) / run_size
    cost_order = c_o * np.mean(total_order) / run_size
    total_costs = cost_lostsales + cost_holding + cost_perish + cost_order
        
    if verbose == True :
        # Print the average cost per time unit
        print("Cost of lost sales:\t", round(cost_lostsales, 4))
        print("Holding cost: \t\t", round(cost_holding, 4))
        print("Perishing cost: \t", round(cost_perish, 4))
        print("Order costs: \t\t", round(cost_order,4))
        print("Total costs: \t\t", round(total_costs,4))
    
    # Return the average total costs
    return total_costs