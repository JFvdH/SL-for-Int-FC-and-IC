# Package imports
import numpy as np
import pandas as pd

# Function definition
def evalModel(input_data, cutoff_date, model, params, verbose = False) :
    
    # Initialize problem parameters
    L_max = params['L_max'] # Maximum leadtime
    m = input_data.iloc[0].loc['m'] # Time to perish 
    policy = params['policy'] # Issuing policy
    c_t = params['c_t'] # Cost of lost sales
    c_h = params['c_h'] # Unit holding cost
    c_o = params['c_o'] # Per unit order cost
    c_p = params['c_p'] # Perishing cost
    
    # Initialize model variables and parameters
    data = input_data.copy()
    data['product_ID'] = data.product_ID.cat.remove_unused_categories()
    start_date = data["date"].min()
    end_date = data["date"].max()
    run_size = len(list(pd.date_range(cutoff_date, end_date, freq='d').astype(str)))-1
    n_runs = data["product_ID"].nunique()
    n_jobs = params['n_jobs'] # Amount of cores used during training
    
    # Initialize the system
    inv = np.zeros(n_runs).astype(int)
    products = data["product_ID"].unique()
    pipeline = (np.zeros(shape=(n_runs, max(0,L_max)))).astype(int) 
    perishline = (np.zeros(shape=(n_runs, m))).astype(int)
    leadtimes = np.zeros(shape=(n_runs, L_max+1))
    for i in range(len(products)) :
        L = data[data['product_ID'] == products[i]].iloc[0]['L']
        leadtimes[i,L] = 1
    open_dates = [str(date)[:10] for date in data[data['store_closed']==0]['date'].unique()]
        
    # Initialize counters
    total_holding = np.zeros(n_runs)
    total_underage = np.zeros(n_runs)
    total_perished = np.zeros(n_runs)
    total_ordered = np.zeros(n_runs)

    # Evaluate the system using simulation
    for i in list(pd.date_range(start_date, end_date, freq='d').astype(str)) :
               
        # Do not count the warmup period
        if i == cutoff_date :
            total_holding = np.zeros(n_runs)
            total_underage = np.zeros(n_runs)
            total_perished = np.zeros(n_runs)
            total_ordered = np.zeros(n_runs)
        
        # Process operations if the store is open
        if i in open_dates :
            
            # Process arrivals
            inv += pipeline[:,0]
            pipeline = pipeline[:,1:].copy()
            
            # Determine order sizes and process expediting arrivals if need be
            inv_input = pd.DataFrame(np.hstack([np.expand_dims(inv,1), pipeline, perishline]),
                                     columns  = ['I'] + ['A_'+str(j) for j in range(1, L_max)]
                                                 + ["P_"+str(i) for i in range(1, m+1)])
            feature_input = data[(data['product_ID']==products[0])&(data['date']==i)].copy().drop(['demand', 'm', 'L', 'store_closed', 'date'], axis=1)
            for j in range(1, len(products)) :
                feature_data = data[(data['product_ID']==products[j])&(data['date']==i)].copy().drop(['demand', 'm', 'L', 'store_closed', 'date'], axis=1)
                if feature_data.shape[0] == 1:
                    feature_input = pd.concat([feature_input, feature_data])
                else :
                    feature_input = pd.concat([feature_input, 
                                               pd.DataFrame(np.full((1, feature_data.shape[1]), np.nan),
                                                            columns = feature_input.columns)])
            model_input = pd.concat([inv_input, feature_input.reset_index(drop=True)], axis=1)
            preds = model.predict(model_input, n_jobs=n_jobs)
            model_order = np.floor(np.maximum(0, preds)).astype(int)

            # Add order to the pipeline 
            pipeline = np.hstack([pipeline, np.zeros(shape=(n_runs,1)).astype(int)])
            for j in range(L_max) :
                pipeline[:,j] += np.multiply(leadtimes[:,j+1], model_order).astype(int)
            
            # Subtract demand
            D = np.zeros(n_runs)
            for j in range(len(products)) :
                entry = data[(data['product_ID']==products[j])&(data['date']==i)]
                D[j] = int(entry['demand'])
                inv[j] -= D[j]
        
            # Update the perishing pipeline w.r.t. demand
            if policy == "FIFO" :
                for j in range(m) :
                    rQty = np.minimum(perishline[:,j], D).astype(int)
                    perishline[:,j] -= rQty
                    D -= rQty
            elif policy == "LIFO" :
                for j in range(m) :
                    rQty = np.minimum(perishline[:,m-1-j], D).astype(int)
                    perishline[:,m-1-j] -= rQty
                    D -= rQty        
            else :
                raise ValueError("Policy not recognized")
        
            # Update the perishing pipeline w.r.t. new arrivals
            perished = perishline[:,0].copy()
            perishline = perishline[:,1:]
            inv -= perished
            arrival = pipeline[:,0].copy()
            perishline = np.hstack([perishline, np.expand_dims(arrival, 1)])
        
            # Register stats and 'lost sales'
            total_holding += np.maximum(0, inv)
            total_underage += np.maximum(0, inv*-1)
            total_perished += perished
            total_ordered += model_order
            inv = np.maximum(0, inv).astype(int)
            
    # Determine the average costs per period per product
    cost_holding = c_h * np.mean(total_holding) / run_size 
    cost_underage = c_t * np.mean(total_underage) / run_size
    cost_perishing = c_p * np.mean(total_perished) / run_size 
    cost_order = c_o * np.mean(total_ordered) / run_size
    total_costs = cost_holding + cost_underage + cost_perishing + cost_order

    # Return the average total costs per period per product
    return total_costs