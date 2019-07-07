import csv 
import numpy as np
import math as m
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mpdf
from mpl_toolkits.mplot3d import Axes3D

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
    train = data[:13105,:]
    test = data[13105:,:]
    # Returns testing data and training data
    return train,test

# This function calculates the average of all samples in the dataset which will be
# used for the principal component analysis algorithm, in another dimension reduction method
# known as linear discriminant analysis we will also use this function to calculate the mean 
# of all smaples that belong to a particular class
def mean(data):
    sum_arr = np.zeros((1,6))
    for i in range(len(data)):
        sum_arr += data[i,:]
    mean = sum_arr/len(data)
    #mean = np.squeeze(mean)
    return mean

# This function calculates the within class scatter matrix and the betwewen class scatter
# matrix which we will use later to find our discriminants
def scatter(data):
    cov_b,cov_w,diff = np.zeros((6,6)),np.zeros((6,6)),np.zeros((6,6))

    zero,one,two,three,four,five = split_data_by_class(data,data)

    zero_mean,one_mean,two_mean,three_mean,four_mean,five_mean = mean(zero),mean(one),mean(two),mean(three),mean(four),mean(five)

    mean_v = [ zero_mean, one_mean, two_mean, three_mean, four_mean, five_mean]
    sample_size = np.array([len(zero),len(one),len(two),len(three),len(four),len(five)]) 
    mean_v = np.array(mean_v)
    mean_o = mean(data[:,1:])
    for avg in range(len(mean_v)):
        for i in range(len(data)):
            var = np.subtract(data[i,1:],mean_v[avg])
            diff += var.T.dot(var)
        cov_w += diff
        cov_b += sample_size[avg]*(mean_v[avg]-mean_o).T.dot(mean_v[avg]-mean_o)
    return cov_w,cov_b

# This program splits preprocessed data by class
def split_data_by_class(data,Z_data):
    zero, one, two, three, four, five = [], [], [], [], [], []
    for i in range(len(data)):
        if data[i,0] == 0:
            zero.append(Z_data[i,1:])
        if data[i,0] == 1:
            one.append(Z_data[i,1:])
        if data[i,0] == 2:
            two.append(Z_data[i,1:])
        if data[i,0] == 3:
            three.append(Z_data[i,1:])
        if data[i,0] == 4:
            four.append(Z_data[i,1:])
        if data[i,0] == 5:
            five.append(Z_data[i,1:])
    zero,one,two,three,four,five = np.array(zero),np.array(one),np.array(two),np.array(three),np.array(four),np.array(five)

    return zero,one,two,three,four,five

def eig(scatter_w,scatter_b):
    eig_val, eig_vec = np.linalg.eig(np.linalg.inv(scatter_w).dot(scatter_b))
    return eig_val,eig_vec

def transform_data(data):
    scatter_w,scatter_b = scatter(data)
    W,V = eig(scatter_w,scatter_b)
    z_data = data[:,1:].dot(V[:,:3])
    return z_data

def euclidean_dist(x1,y1,z1,x2,y2,z2):
   return m.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2) 

def mode(array):
    count = 0
    mode_count = 0
    mode = 0
    for i in range(len(array)-1):
        for j in range(i+1,len(array)):
            if array[i] == array[j]:
                count += 1
        if count > mode_count:
            mode_count = count
            mode = array[i]
    return mode

def knn(x,train_x,train_y):
    pred_y = []
    for i in range(len(x)):
        distance = []
        for j in range(len(train_x)):
            dist = [euclidean_dist(x[i,0],x[i,1],x[i,2],train_x[j,0],train_x[j,1],train_x[j,2]), train_y[j]]
            distance.append(dist)
        distance = sorted(distance,key=lambda l:l[0],reverse=False)
        distance = np.array(distance)
        k_classes = distance[:3,1]
        prediction = mode(k_classes)
        pred_y.append(prediction)
    return pred_y
def score(pred_y,y):
    count = 0;
    n = len(y)
    for i in range(n):
        if pred_y[i] != y[i]:
            count += 1
    return count/n*100
data = load_data("falldetection.csv")
z_test_data,z_train_data = transform_data(data[13105:,:]),transform_data(data[:13105,:])
#print(z_data)
pred = knn(z_test_data,z_train_data,data[:13105,0])
print("Accuracy : ",score(pred,data[13105:,0]))
