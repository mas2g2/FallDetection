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

def mean(data):
    sum_arr = np.zeros((1,6))
    for i in range(16382):
        sum_arr += data[i,1:]
    mean = sum_arr/16382
    return mean

def cov(data,mean):
    diff = np.zeros((6,6))
    for i in range(len(data)):
        diff += (data[i,1:] - mean).T.dot(data[i,1:] - mean)
    cov = diff/16382
    return cov

data = load_data("falldetection.csv");
shape = data.shape
mean = mean(data)
cov = cov(data,mean)
print("Mean: ",mean)
print("Covariance: ",cov)
w,v = np.linalg.eig(cov)
print("Weights :\n",w);
print("Vectors : \n",v)
print("Proportion of variance with first three eig val : ",np.sum(w[:3])/np.sum(w))
print(w[:3])
W = v[:,:3]
print(W)
#train,test = split_data(data,train=0.8,test=0.2)

