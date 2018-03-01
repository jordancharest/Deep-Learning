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
    input_train.append(training_set_scaled[i-timesteps:i, 0])   # contains Open stock price from the previous timesteps
    output_train.append(training_set_scaled[i, 0])  # contains current Open stock price
    
input_train = np.array(input_train)
output_train = np.array(output_train)

# Reshape numpy array to add more dimensions (for example if you want to factor in the Close stock price)
#                               shape (length) along dimension 0
#                                           |
#                                           |
input_train = np.reshape(input_train, (input_train.shape[0], timesteps, 1))  # (batch_size, timesteps, input_dims)
#                                                                  |
#                                                                  |
#                                     keep input size at 1 since we aren't adding any indicators right now



## BUILDING THE RNN ===============================================================================
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialize
regressor = Sequential()

# Add LSTM Layers with Dropout Regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps,1)))
regressor.add(Dropout(0.2)) # randomly ignore 20% of neurons in every epoch (prevent overfitting)

regressor.add(LSTM(units=50, return_sequences=True))    # input shape is automatically deduced after first layer
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=False))   # only returning one value to output layer (return_sequences=False)
regressor.add(Dropout(0.2))

# Add Output Layer
regressor.add(Dense(units=1))

# Compile
regressor.compile(optimizer='adam', loss='mean_squared_error')  # RMSprop is usually a good choice for RNNs (adam is used here on suggestion from Hadelin)


# Fit RNN to the training set
regressor.fit(input_train, output_train, epochs=100, batch_size=32) # batch_size = how many input data points before back propagation


## MAKING PREDICTIONS =============================================================================
raw_test_data = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = raw_test_data.iloc[:, 1:2].values

# Predict the stock price
total_dataset = pd.concat((raw_training_data['Open'], raw_test_data['Open']), axis=0)   # verical concatenation
inputs = total_dataset[len(total_dataset)-len(raw_test_data)-60: ].values
inputs = inputs.reshape(-1,1)
inputs = scale.transform(inputs)

input_test = []

for i in range(timesteps, len(raw_test_data)+timesteps ):
    input_test.append(inputs[i-timesteps:i, 0])   # contains Open stock price from the previous timesteps
        
input_test = np.array(input_test)
input_test = np.reshape(input_test, (input_test.shape[0], timesteps, 1))  # (batch_size, timesteps, input_dims)

predicted_stock_price = regressor.predict(input_test)
predicted_stock_price = scale.inverse_transform(predicted_stock_price)

# Visualize the results
plt.plot(real_stock_price, color='red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color='blue', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
