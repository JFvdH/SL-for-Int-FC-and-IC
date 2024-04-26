# Package imports
import copy
import pandas as pd
import numpy as np

# The augmentData function is used to augment the data with inventory and arrival data
# Takes as input the cleaned feature data
# Outputs the same data augmented with inventory and arrival columns
def augmentData(input_data, params, hyperparams) :
    
    # Retrieve relevant (hyper)parameter information
    L_max = params["L_max"] # Leadtime
    last_date = str(input_data['date'].max())[:10]
    D_index = params["D_index"] # Index of the column containing demand information
    train_size = params["train_size"] # Number of desired training samples
    inv_multiplier = hyperparams['inv_multiplier'] # Hyperparameter used for inventory column generation
    
    # Obtain desired demand characteristics
    data = input_data[input_data['store_closed']==0].copy()
    products = data["product_ID"].unique()
    product_means = {}
    product_stds = {}
    for product_ID in products :
        product_means[product_ID] = data[data["product_ID"] == product_ID]["demand"].mean()
        product_stds[product_ID] = data[data["product_ID"] == product_ID]["demand"].var()
    D = np.expand_dims(data.iloc[:,D_index], 1)
    for i in range(1, L_max+1) :
        nextD = np.expand_dims(np.hstack([data.iloc[i:,D_index], np.full(i, np.nan)]), 1)
        D = np.hstack([D, nextD])
    D = pd.DataFrame(D, columns=["D_"+str(i) for i in range(1, L_max+2)])
    cData = pd.concat([data.reset_index(drop=True), D.reset_index(drop=True)], axis=1)
    cData.drop(cData.loc[pd.to_timedelta(cData['L'], unit='D') + cData["date"] >= last_date].index, inplace=True)
    for i in range(L_max+1) :
        for j in range(i+2, L_max+2) :
            cData.loc[cData['L']==i, 'D_'+str(j)] = 0
        
    # Extend the dataset to the desired training size
    aData = copy.deepcopy(cData)
    for i in range(int(train_size/cData.shape[0])-1) :
        aData = pd.concat([aData, cData], axis=0)
    indices = np.random.randint(cData.shape[0], size=train_size-cData.shape[0]*int(train_size/cData.shape[0]))
    aData = pd.concat([aData, aData.iloc[indices,:]], axis=0)
    aData.reset_index(inplace=True, drop=True)
    mean = np.zeros(aData.shape[0])
    std = np.zeros(aData.shape[0])
    for product_ID in products :
        mean += (aData["product_ID"] == product_ID) * product_means[product_ID] 
        std += (aData["product_ID"] == product_ID) * product_stds[product_ID] 
    
    # Generate the relevant columns    
    I = np.random.randint(0, mean*inv_multiplier+1, size = aData.shape[0])
    A = np.random.randint(0, mean*2+1, size = (aData.shape[0], 1))
    for i in range(1, L_max-1) :
        A = np.hstack([A, np.random.randint(0, mean*2+1, size = (aData.shape[0], 1))])
    augmentation = pd.DataFrame(np.hstack([np.expand_dims(I,1), A]),
                                columns=["I"] + ["A_"+str(i) for i in range(1, L_max)])
    
    # Augment the data
    aData = pd.concat([augmentation, aData], axis=1)
    
    # Set arrivals to 0 when outside of leadtime bounds
    for i in range(L_max+1) :
        for j in range(max(1,i), L_max) :
            aData.loc[aData['L']==i, 'A_'+str(j)] = 0
                
    # Return the augmented dataset
    return aData