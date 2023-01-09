# Package imports
import numpy as np

# Function definition
def evalModel(model_exp, model_reg, params, indep = False, verbose = False) :
    
    # Initialize problem parameters
    h = params['h'] # Unit holding cost
    b = params['b'] # Unit backorder cost
    c_r = params['c_r'] # Regular unit order cost
    c_e = params['c_e'] # Expediting unit order cost
    l_r = params['l_r'] # Regular leadtime
    l_e = params['l_e'] # Expediting  # TODO: Make the implementation work for l_e >  0
    min_demand = params['min_demand'] # Minimum amount of demand in a period
    max_demand = params['max_demand'] # Maximum amount of demand in a period
    
    # Initialize model parameters
    test_size = params['test_size'] # Amount of sample data used for evaluation
    run_size = params['run_size'] # Amount of sample data per evaluation run
    warmup = params['warmup'] # Amount of sample data used for the evaluation warmup period
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Derive the number of runs to execute
    n_runs = round(test_size/run_size)
    
    # Initialize the system
    demand = np.random.randint(min_demand, max_demand+1, size=(n_runs, run_size+warmup))
    inv = (np.ones(n_runs)*((max_demand+min_demand)/2)*l_e).astype(int)
    pipeline_reg = (np.zeros(shape=(n_runs, l_r-1))).astype(int)
    pipeline_exp = (np.zeros(shape=(n_runs, max(0,l_e-1)))).astype(int)
        
    # Initialize counters
    total_underage = np.zeros(n_runs)
    total_holding = np.zeros(n_runs)
    total_acq_exp = np.zeros(n_runs)
    total_acq_reg = np.zeros(n_runs)
    
    # Evaluate the system using simulation
    for i in range(run_size+warmup) :
        
        # Do not count the warmup period
        if i == warmup :
            total_underage = np.zeros(n_runs)
            total_holding = np.zeros(n_runs)
            total_acq_exp = np.zeros(n_runs)
            total_acq_reg = np.zeros(n_runs)
        
        # Determine order sizes and process expediting arrivals if need be
        exp_input = np.hstack([np.expand_dims(inv,1), pipeline_exp])
        exp_order = np.maximum(0, np.round(model_exp.predict(exp_input, n_jobs=n_jobs))).astype(int)
        if indep == False :
            reg_input = np.hstack([np.expand_dims(inv+exp_order,1), pipeline_reg])
        else :
            reg_input = np.hstack([np.expand_dims(inv,1), pipeline_reg])
        reg_order = np.maximum(0, np.round(model_reg.predict(reg_input, n_jobs=n_jobs))).astype(int)
        if l_e == 0 :
            inv += exp_order
        else :
            pipeline_exp = np.hstack([pipeline_exp, exp_order])
        pipeline_reg = np.hstack([pipeline_reg, np.expand_dims(reg_order,1)])
        inv -= demand[:,i]
        
        # Register stats
        total_acq_exp += exp_order
        total_acq_reg += reg_order
        total_holding += np.maximum(0, inv)
        total_underage += np.maximum(0, inv*-1)
        
        # Process arrivals
        if l_e != 0 :
            inv += pipeline_exp[:,0]
            pipeline_exp = pipeline_exp[:,1:].copy()
        inv += pipeline_reg[:,0]
        pipeline_reg = pipeline_reg[:,1:].copy()
        
    # Determine the average cost per time unit    
    cost_underage = b * np.mean(total_underage) / run_size
    cost_holding = h * np.mean(total_holding) / run_size
    cost_order_exp = c_e * np.mean(total_acq_exp) / run_size
    cost_order_reg = c_r * np.mean(total_acq_reg) / run_size
    total_costs = cost_underage + cost_holding + cost_order_exp + cost_order_reg
    
    if verbose == True :
        # Print the average cost per time unit
        print("Backorder costs:\t\t\t", round(cost_underage, 4))
        print("Holding costs: \t\t\t\t", round(cost_holding, 4))
        print("Expediting order costs: \t", round(cost_order_exp, 4))
        print("Regular order costs: \t\t", round(cost_order_reg,4))
        print("Total costs: \t\t\t\t", round(total_costs,4))
    
    # Return the average total costs
    return total_costs
        