import csv 
import numpy as np
import math as m
import matplotlib.pyplot as plt
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

# This function calculates the average of all samples in the dataset which will be
# used for the principal component analysis algorithm
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
print("Proportion of variance with first three eig val : ",np.sum(w[:2])/np.sum(w))

W = v[:,:2]

train,test = split_data(data,train=0.8,test=0.2)
train_X,train_Y = train[:,1:],train[:,0]

z_train = W.T.dot(train_X.T)
x = z_train[0,:]
y = z_train[1,:]
plt.plot(x,y)
plt.show()
print(z_train.shape)

