import csv 
import numpy as np
import math as m

# This function loads data from csv file to numpy array
def load_data(filename):
    # Opens file for reading
    with open(filename) as f:
        # Passes contents of csv file to list
        data = list(csv.reader(f));
        # Converts list to array
        data = np.array(data)
    # Removes headers
    data = data[1:,:]
    # Turns array into array of float values
    data = data.astype(float)
    return data

# This function splits data into training and testing data
def split_data(data,train=0.5,test=0.5):
    # Gets size of training data
    train_len = m.floor(data.shape[0]*train)
    # Gets size of testing data
    test_len = data.shape[0] - train_len
    train = data[:train_len,:]
    test = data[train_len,:]
    # Returns testing data and training data
    return train,test
    
data = load_data("falldetection.csv");
shape = data.shape
print(shape[0])
train,test = split_data(data,train=0.8,test=0.2)
print(train)
