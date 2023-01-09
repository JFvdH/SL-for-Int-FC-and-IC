# Package imports
import numpy as np

# Function definition
# Takes params as input and yields an X and Y to train on as output
def generateData(params, hyperparams) :
    
    # Initialize problem parameters
    m = params['m'] # Time to perish 
    L = params['L'] # Lead time 
    mu = params['mu'] # Mean
    sigma = params['sigma'] # Standard deviation
    
    # Initialize model parameters
    train_size = params['train_size'] # Amount of sample data used for training
    inv_multiplier = hyperparams['inv_multiplier'] # Specifies inventory range over which data is generated

    # Use parameters to derive key variables
    a = mu**2/sigma**2 # Shape parameter of the gammma distribution
    scale = sigma**2/mu # Scale parameter of the gamma distribution
    
    # Randomly generate possible situations
    I = np.random.randint(0, inv_multiplier*mu+1, size=(train_size))
    D = np.random.gamma(a, scale, size=(L+m, train_size)).round().astype(int)
    A = np.random.gamma(a, scale, size=(L-1, train_size)).round().astype(int)
    perishables = I.copy()
    P = np.zeros(shape=(m, train_size)).astype(int)
    for i in range(m-1) :
        P[i,:] = np.random.binomial(perishables, 1/(m-i))
        perishables -= P[i,:]
    P[m-1,:] = perishables
        
    # Return relevant variables
    return (I,      # Inventory at the start of a period t, after arrivals
            D,      # Demand during a period t
            A,      # Arrivals between order point and delivery point
            P)      # Expiry moments and quantities at moment of arrival
    
        
    