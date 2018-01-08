# Artificial Neural Network


# Data Preprocessing

import numpy as np
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Churn_Modeling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data (country and gender)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:, 1] = labelencoder_country.fit_transform(X[:, 1])
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

# Split Country data into different columns
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Avoid Dummy Variable Trap
X = X[:, 1:]

# Split the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Make the Neural Network

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Add the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Add the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Make predictions and evaluating the model

# Predict the Test set results
y_pred = classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)


# Make a new prediction for a single customer
customer = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
customer = sc.transform(customer)

new_prediction = classifier.predict(customer)

if new_prediction > 0.5:
    print('The customer is expected to leave the bank')
else:
    print('The customer is expected to stay with the bank')
    
    
      
# Evaluate the performance of the Neural Network using k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

# Improve the ANN
# Dropout Regularization to reduce overfitting if needed (not needed on this model, low variance of accuracy)
# Use this line below (add after each layer you want to add Dropout too)
# classifier.add(Dropout(p = 0.1))


# Tune the ANN using Grid Search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_