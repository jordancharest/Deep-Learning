# Recurrent Neural Network

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## DATA PREPROCESSING =============================================================================
# Import the training set
raw_training_data = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = raw_training_data.iloc[:, 1:2].values

# Feature Scaling - recommended to use normalization over standardization for RNNs
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)

# Create the "Memory Cell" Data Structure
# Look <timesteps> in the past and produce 1 output (tomorrow's stock price)
input_train = []
output_train = []
timesteps = 60

for i in range(timesteps, len(training_set_scaled) ):
    # contains Open stock price from the previous timesteps
    input_train.append(training_set_scaled[i-timesteps:i, 0])
    # contains current Open stock price
    output_train.append(training_set_scaled[i, 0])
    
input_train = np.array(input_train)
output_train = np.array(output_train)

# Reshape numpy array to add more dimensions (for example if you want to factor in the Close stock price)
#                               shape (length) along dimension 0
#                                           |
#                                           |
input_train.reshape(input_train, (input_train.shape[0], timesteps, 1))  # (batch_size, timesteps, input_dims)
#                                                                  |
#                                                                  |
#                                     keep input size at 1 since we aren't adding any indicators right now

## BUILDING THE RNN ===============================================================================