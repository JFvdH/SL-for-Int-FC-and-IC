# Package imports
import numpy as np
import pandas as pd

# Function definition
def evalModel(data, model, params, length_timeseries = 365, cutoff_date = 90) :

    # Initialize problem parameters
    L = params['L'] # Maximum leadtime
    h = params['h'] # Unit holding cost
    p = params['p'] # Unit penalty cost
    
    # Initialize model variables and parameters
    run_size = length_timeseries
    n_runs = data["Series"].nunique()
    n_jobs = params['n_jobs'] # Amount of cores used during prediction
    independent_variables = [var for var in data.columns if var not in ["D_1", "Series", "Day"]]

    # Initialize the system
    inv = np.zeros(n_runs).astype(int)
    pipeline = (np.zeros(shape=(n_runs, max(0,L)))).astype(int) 

    # Initialize counters
    total_holding = np.zeros(n_runs)
    total_underage = np.zeros(n_runs)

    # Evaluate the system using simulation
    for i in range(run_size) :
        
        # Do not count the warmup period
        if i == cutoff_date :
            total_holding = np.zeros(n_runs)
            total_underage = np.zeros(n_runs)
                 
        # Process arrivals
        if L != 0 :
            inv += pipeline[:,0]
            pipeline = pipeline[:,1:].copy()
        
        # Determine order sizes and process expediting arrivals if need be
        inv_input = pd.DataFrame(np.hstack([np.expand_dims(inv,1), pipeline]),
                                 columns  = ['I'] + ['A_'+str(j) for j in range(1, L)])
        feature_input = data[data['Day']==i][independent_variables]
        model_input = pd.concat([inv_input, feature_input.reset_index(drop=True)], 
                                axis=1, copy=False)
        preds = model.predict(model_input, n_jobs=n_jobs)
        model_order = np.ceil(np.maximum(0, preds)).astype(int)
        
        # Add order to inventory immediately if leadtime equals 0
        if L == 0 :
            inv += model_order
        # Otherwise, add order to the pipeline
        else :
            pipeline = np.hstack([pipeline, model_order])

        # Subtract demand
        inv -= data[data['Day']==i]['D_1'].to_numpy().astype(int)
    
        # Register stats and 'lost sales'
        total_holding += np.maximum(0, inv)
        total_underage += np.maximum(0, inv*-1)
        inv = np.maximum(0, inv).astype(int)
            
    # Determine the average costs per period per run
    cost_holding = h * np.mean(total_holding) / run_size 
    cost_underage = p * np.mean(total_underage) / run_size
    total_costs = cost_holding + cost_underage

    # Return the average total costs per period per run
    return total_costs