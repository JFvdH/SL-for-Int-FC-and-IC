# Package imports
import numpy as np

# Function definition
def generateData(params, hyperparams) :

    # Initialize problem parameters
    distribution = params['distribution'] # Distribution followed by the demand
    mean = params['mean'] # Expected demand
    L = params['L'] # Leadtime
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for training
    inv_multiplier = hyperparams['inv_multiplier'] # Specifies inventory range over which data is generated
    
    # Randomly generate possible situations
    I = np.random.randint(0, inv_multiplier*mean+1, size=(train_size))
    if distribution == 'Poisson' :
        D = np.random.poisson(mean, size=(L+1, train_size))
        A = np.random.poisson(mean, size=(L-1, train_size))
    elif distribution == 'geometric' : 
        D = np.random.geometric(1/(mean+1), size=(L+1, train_size))
        A = np.random.geometric(1/(mean+1), size=(L-1, train_size))    
    else :
        raise ValueError("The following distribution is not recognized:", distribution)
        
    # Return relevant variables
    return (I,      # Inventory at the start of a period t, after arrivals
            D,      # Demand during a period t
            A)      # Arrivals between order point and delivery point