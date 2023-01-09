# Package imports
import numpy as np

# Function definition
def generateData(params, hyperparams):
    
    # Initialize problem parameters
    reg = params['reg'] # Indicator variable of whether the model is a regular model (rather than expediting)
    if reg == False :
        l = params['l_e'] 
        inv_multiplier = hyperparams['inv_multiplier_1'] 
    else :
        l = params['l_r']
        inv_multiplier = hyperparams['inv_multiplier_2']
    min_demand = params['min_demand'] # Minimum amount of demand in a period
    max_demand = params['max_demand'] # Maximum amount of demand in a period
    mean = int((min_demand+max_demand)/2)
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for data generation

    # Demand and noise generation 
    I = np.random.randint(-1*inv_multiplier*mean, inv_multiplier*mean+1, size = train_size)
    D = np.random.randint(min_demand, max_demand+1, size = train_size+l+1)
    A = np.random.randint(min_demand, max_demand+1, size = train_size+l+1)

    # Return relevant variables
    return (I,      # Inventory at the start of a period t
            D,      # Demand during a period t
            A)      # Arrivals at the end of a period t