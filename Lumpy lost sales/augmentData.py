# Package imports
import copy
import pandas as pd
import numpy as np

# The augmentData function is used to augment the data with inventory and arrival data
# Takes as input the cleaned feature data
# Outputs the same data augmented with inventory and arrival columns
def augmentData(input_data, params, hyperparams) :
    
    # Retrieve relevant (hyper)parameter information
    L = params["L"] # Leadtime
    train_size = params["train_size"] # Number of desired training samples
    inv_multiplier = hyperparams['inv_multiplier'] # Hyperparameter used for inventory column generation
    
    # Extend the dataset to the desired training size
    aData = copy.deepcopy(input_data)
    for i in range(int(train_size/input_data.shape[0])-1) :
        aData = pd.concat([aData, input_data], axis=0)
    indices = np.random.randint(input_data.shape[0], size=train_size-input_data.shape[0]*int(train_size/input_data.shape[0]))
    aData = pd.concat([aData, aData.iloc[indices,:]], axis=0)
    aData.reset_index(inplace=True, drop=True)
    mean = aData["D_1"].mean()
    
    # Generate the relevant columns    
    I = np.random.randint(0, mean*inv_multiplier+1, size = aData.shape[0])
    A = np.random.randint(0, mean*2+1, size = (aData.shape[0], L))
    for i in range(1, L-1) :
        A = np.hstack([A, np.random.randint(0, mean*2+1, size = (aData.shape[0], L))])
    augmentation = pd.DataFrame(np.hstack([np.expand_dims(I,1), A]),
                                columns=["I"] + ["A_"+str(i) for i in range(1, L)])
    
    # Augment the data
    aData = pd.concat([augmentation, aData], axis=1, copy=False)
                
    # Return the augmented dataset
    return aData