# Package imports
import numpy as np
import pandas as pd

# Function definition
def evalModel(input_data, cutoff_date, model_exp, model_reg, params, verbose = False) :
    
    # Initialize problem parameters
    L_max = params['L_max'] # Maximum leadtime
    h = params['h'] # Unit holding cost
    b = params['b'] # Unit penalty cost
    c_r = params['c_r'] # Regular unit order cost
    c_e = params['c_e'] # Expediting unit order cost
    
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
    leadtimes = np.zeros(shape=(n_runs, L_max+1))
    for i in range(len(products)) :
        L = data[data['product_ID'] == products[i]].iloc[0]['l_r']
        leadtimes[i,L] = 1
    open_dates = [str(date)[:10] for date in data[data['store_closed']==0]['date'].unique()]
        
    # Initialize counters
    total_holding = np.zeros(n_runs)
    total_underage = np.zeros(n_runs)
    total_exp_order = np.zeros(n_runs)
    total_reg_order = np.zeros(n_runs)

    # Evaluate the system using simulation
    for i in list(pd.date_range(start_date, end_date, freq='d').astype(str)) :
               
        # Do not count the warmup period
        if i == cutoff_date :
            total_holding = np.zeros(n_runs)
            total_underage = np.zeros(n_runs)
            total_exp_order = np.zeros(n_runs)
            total_reg_order = np.zeros(n_runs)
        
        # Process operations if the store is open
        if i in open_dates :
            
            # Process arrivals
            inv += pipeline[:,0]
            pipeline = pipeline[:,1:].copy()
            
            # Determine expediting order sizes and process expediting arrivals if need be
            inv_exp_input = pd.DataFrame(np.expand_dims(inv,1),
                                     columns  = ['I_exp'])
            feature_input = data[(data['product_ID']==products[0])&(data['date']==i)].drop(['demand', 'l_r', 'store_closed', 'date'], axis=1)
            for j in range(1, len(products)) :
                feature_data = data[(data['product_ID']==products[j])&(data['date']==i)].drop(['demand', 'l_r', 'store_closed', 'date'], axis=1)
                if feature_data.shape[0] == 1:
                    feature_input = pd.concat([feature_input, feature_data])
                else :
                    feature_input = pd.concat([feature_input, 
                                               pd.DataFrame(np.full((1, feature_data.shape[1]), np.nan),
                                                            columns = feature_input.columns)])
            model_exp_input = pd.concat([inv_exp_input, feature_input.reset_index(drop=True)], axis=1)
            preds_exp = model_exp.predict(model_exp_input, n_jobs=n_jobs)
            model_order_exp = np.round(np.maximum(0, preds_exp)).astype(int)
            inv += model_order_exp
            
            # Determine regular order sizes
            inv_reg_input = pd.DataFrame(np.hstack([np.expand_dims(inv,1), pipeline]),
                                     columns  = ['I_reg'] + ['A_'+str(j) for j in range(1, L_max)])
            model_reg_input = pd.concat([inv_reg_input, feature_input.reset_index(drop=True)], axis=1)
            preds_reg = model_reg.predict(model_reg_input, n_jobs=n_jobs)
            model_order_reg = np.round(np.maximum(0, preds_reg)).astype(int)
            
            # Add order to inventory immediately if leadtime equals 0
            inv += np.multiply(leadtimes[:,0], model_order_reg).astype(int)

            # Add order to the pipeline otherwise
            pipeline = np.hstack([pipeline, np.zeros(shape=(n_runs,1)).astype(int)])
            for j in range(L_max) :
                pipeline[:,j] += np.multiply(leadtimes[:,j+1], model_order_reg).astype(int)
            
            # Subtract demand
            for j in range(len(products)) :
                entry = data[(data['product_ID']==products[j])&(data['date']==i)]
                inv[j] -= int(entry['demand'])
        
            # Register stats and 'lost sales'
            total_holding += np.maximum(0, inv)
            total_underage += np.maximum(0, inv*-1)
            total_exp_order += model_order_exp
            total_reg_order += model_order_reg
            
    # Determine the average costs per period per product
    cost_holding = h * np.mean(total_holding) / run_size 
    cost_underage = b * np.mean(total_underage) / run_size
    cost_exp_order = c_e * np.mean(total_exp_order) / run_size
    cost_reg_order = c_r * np.mean(total_reg_order) / run_size
    total_costs = cost_holding + cost_underage + cost_exp_order + cost_reg_order

    # Return the average total costs per period per product
    return total_costs