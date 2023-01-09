# Package imports
import numpy as np

# Function definition
def evalModel(model, params, verbose = False) :
    
    # Initialize problem parameters
    distribution = params['distribution'] # Distribution followed by the demand
    mean = params['mean'] # Expected demand
    L = params['L'] # Leadtime
    h = params['h'] # Unit holding cost
    p = params['p'] # Unit penalty cost
    
    # Initialize model parameters
    test_size = params['test_size'] # Amount of sample data used for evaluation
    run_size = params['run_size'] # Amount of sample data per evaluation run
    warmup = params['warmup'] # Amount of sample data used for the evaluation warmup period
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Derive the number of runs to execute
    n_runs = round(test_size/run_size)
    
    # Initialize the system
    if distribution == 'Poisson' :
        demand = np.random.poisson(mean, size=(n_runs, run_size+warmup))
    elif distribution == 'geometric' : 
        demand = np.random.geometric(1/(mean+1), size=(n_runs, run_size+warmup))
        #demand = np.random.geometric(1/mean, size=(n_runs, run_size+warmup))
    else :
        raise ValueError("The following distribution is not recognized:", distribution)
    inv = np.random.randint(0, 2*mean+1, size=n_runs)
    pipeline = (np.zeros(shape=(n_runs, max(0,L-1)))).astype(int)

    # Initialize counters
    total_holding = np.zeros(n_runs)
    total_underage = np.zeros(n_runs)

    # Evaluate the system using simulation
    for i in range(run_size+warmup) :
        
        # Do not count the warmup period
        if i == warmup :
            total_holding = np.zeros(n_runs)
            total_underage = np.zeros(n_runs)
            
        # Determine order sizes and process expediting arrivals if need be
        model_input = np.hstack([np.expand_dims(inv,1), pipeline])
        model_order = np.maximum(0, np.ceil(model.predict(model_input, n_jobs=n_jobs))).astype(int)
        if L == 0 :
            inv += model_order
        else :
            pipeline = np.hstack([pipeline, np.expand_dims(model_order, 1)])
        inv -= demand[:,i]
        
        # Register stats and 'lose sales'
        total_holding += np.maximum(0, inv)
        total_underage += np.maximum(0, inv*-1)
        inv = np.maximum(0, inv).astype(int)
        
        # Process arrivals
        if L != 0 :
            inv += pipeline[:,0]
            pipeline = pipeline[:,1:].copy()
    
    # Determine the average cost per time unit    
    cost_holding = h * np.mean(total_holding) / run_size
    cost_underage = p * np.mean(total_underage) / run_size
    total_costs = cost_holding + cost_underage
    
    if verbose == True :
        # Print the average cost per time unit
        print("Holding costs: \t\t\t", round(cost_holding, 4))
        print("Penalty costs:\t\t\t", round(cost_underage, 4))
        print("Total costs: \t\t\t", round(total_costs,4))
    
    # Return the average total costs
    return total_costs