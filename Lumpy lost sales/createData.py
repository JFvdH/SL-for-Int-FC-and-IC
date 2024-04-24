# Package imports
import numpy as np
import pandas as pd

# Function used to generate train/validation/test time series data
def createData(n_timeseries = 1000, length_timeseries = 365, days_features = 14, frequency = 2) :
        
    # Creating the time series: Noise factor and constant
    arr_train = np.zeros((int(n_timeseries*0.6), length_timeseries + days_features)) # 60% training data
    arr_val = np.zeros((int(n_timeseries*0.2), length_timeseries + days_features)) # 20% validation data
    arr_test = np.zeros((int(n_timeseries*0.2), length_timeseries + days_features)) # 20% test data
    
    # Creating the time series: Trend
    for day_index in range(length_timeseries + days_features) :
        if day_index % frequency == 0 :
            for timeseries_index in range(int(n_timeseries*0.6)) : 
                arr_train[timeseries_index, day_index] += np.random.poisson(5)
            for timeseries_index in range(int(n_timeseries*0.2)) :            
                arr_val[timeseries_index, day_index] += np.random.poisson(5)
                arr_test[timeseries_index, day_index] += np.random.poisson(5)

    # Creating the data
    data_train = np.empty(shape=(int(n_timeseries*0.6) * length_timeseries, days_features+4))
    data_val = np.empty(shape=(int(n_timeseries*0.2) * length_timeseries, days_features+4))
    data_test = np.empty(shape=(int(n_timeseries*0.2) * length_timeseries, days_features+4))
    for day_index in range(length_timeseries) :
        
        # For each timeseries_index, create train/validation/test data as follows:
        # - <days_features>+1 features for demand and most recent historical sales respectively
        # - 1 feature for the average sales over the <days_features> most recent historical sales
        # - 1 feature to indicate the time series
        # - 1 feature to indicate the day in the time series
        for timeseries_index in range(int(n_timeseries*0.6)) : 
            data_train[timeseries_index*length_timeseries+day_index,:days_features+1] = arr_train[timeseries_index, day_index:day_index+days_features+1]
            data_train[timeseries_index*length_timeseries+day_index,days_features+1] = arr_train[timeseries_index, day_index:day_index+days_features+1].mean()
            data_train[timeseries_index*length_timeseries+day_index,days_features+2] = timeseries_index
            data_train[timeseries_index*length_timeseries+day_index,days_features+3] = day_index           
        for timeseries_index in range(int(n_timeseries*0.2)) : 
            data_val[timeseries_index*length_timeseries+day_index,:days_features+1] = arr_val[timeseries_index, day_index:day_index+days_features+1]
            data_val[timeseries_index*length_timeseries+day_index,days_features+1] = arr_val[timeseries_index, day_index:day_index+days_features+1].mean()
            data_val[timeseries_index*length_timeseries+day_index,days_features+2] = timeseries_index
            data_val[timeseries_index*length_timeseries+day_index,days_features+3] = day_index    
            data_test[timeseries_index*length_timeseries+day_index,:days_features+1] = arr_test[timeseries_index, day_index:day_index+days_features+1]
            data_test[timeseries_index*length_timeseries+day_index,days_features+1] = arr_test[timeseries_index, day_index:day_index+days_features+1].mean()
            data_test[timeseries_index*length_timeseries+day_index,days_features+2] = timeseries_index
            data_test[timeseries_index*length_timeseries+day_index,days_features+3] = day_index

    # Creating the dataframes
    df_train = pd.DataFrame(data_train, columns = ["D_1"] + ["day_"+str(i) for i in range(1, days_features+1)
                                                             ] + ["Avg.", "Series", "Day"]).astype(int)
    df_val = pd.DataFrame(data_val, columns = ["D_1"] + ["day_"+str(i) for i in range(1, days_features+1)
                                                             ] + ["Avg.", "Series", "Day"]).astype(int)
    df_test = pd.DataFrame(data_test, columns = ["D_1"] + ["day_"+str(i) for i in range(1, days_features+1)
                                                             ] + ["Avg.", "Series", "Day"]).astype(int)
    
    # Returning the data
    return df_train, df_val, df_test