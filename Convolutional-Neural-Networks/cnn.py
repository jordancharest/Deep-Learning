# Convolutional Neural Network

## BUILD THE CNN ==================================================================================
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64,64,3), activation='relu'))

# Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1)) # softmax activation if more than two categories

# Compile
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   # categorical cross entropy for multiple categories
