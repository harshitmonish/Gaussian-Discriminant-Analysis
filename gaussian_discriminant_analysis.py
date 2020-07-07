# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:40:42 2020

@author: harshitm
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd

#function to read the input X data in numpy array
def modify_input(x):
    x_f = []
    for i in x:
        temp = str(i[0]).split(" ")
        x_f.append([float(temp[0]), float(temp[2])])
        
    return np.array(x_f)

#function to normalize the input data.
def normalize(x): 
    x= x.T
    meanX1 = np.mean(x[0])
    varX1 = np.std(x[0])
    meanX2 = np.mean(x[1])
    varX2 = np.std(x[1])
    x[0] = (x[0] - meanX1) / varX1
    x[1] = (x[1] - meanX2) / varX2
    return x.T

#fucntion to compute the mean vectors, covariance matrix, and bernoulli param for linear seperator
def compute_linear_parameters(x, y):
    a, b = x[np.where(y == "Alaska")[0]].T
    c, d = x[np.where(y == "Canada")[0]].T
    m = x.shape[0]
    phi = a.shape[0]/m
    mu0 = np.array([np.mean(a), np.mean(b)])
    mu1 = np.array([np.mean(c), np.mean(d)])
    e = np.r_[np.c_[a, b] - mu0, np.c_[c, d] - mu1]
    sigma = np.dot(e.T, e)/m
    return (phi, mu0, mu1, sigma)

#fucntion to compute the mean vectors, covariance matrix, and bernoulli param for quadratic seperator
def compute_quadratic_parameters(x, y):
    a, b = x[np.where(y == "Alaska")[0]].T
    c, d = x[np.where(y == "Canada")[0]].T
    m = x.shape[0]
    phi = a.shape[0]/m
    mu0 = np.array([np.mean(a), np.mean(b)])
    mu1 = np.array([np.mean(c), np.mean(d)])
    e = np.c_[a, b] - mu0
    sigma0 = np.dot(e.T, e)/a.shape[0]
    e = np.c_[c, d] - mu1
    sigma1 = np.dot(e.T, e)/c.shape[0]
    
    return (phi, mu0, mu1, sigma0, sigma1)

#function to compute linear seperator equation cofficient
def compute_linear_boundary(phi, mu0, mu1, sigma, x):
    thetas = np.zeros(3)
    sigmainv = np.linalg.inv(sigma)
    thetas[1:3] = np.dot(mu0.T - mu1.T, sigmainv)
    thetas[0] = np.log((1 - phi)/phi) - mu0.T.dot(sigmainv).dot(mu0)/2 + mu1.T.dot(sigmainv).dot(mu1)/2
    return thetas

#functon to plot the decision boundaries and input data
def plot_graph(x, y, phi, mu0, mu1, sigma0, sigma1, thetas, type_graph):
    a, b = x[np.where(y == "Alaska")[0]].T
    c, d = x[np.where(y == "Canada")[0]].T
    fig = plt.figure(figsize=(10, 8))
    
    #plot the input train data
    plt.scatter(a, b, label='Alaska', marker="+")
    plt.scatter(c, d, label='Canada', marker="x")
    
    #50 is the size of the mesh
    xt = np.linspace(-3, 3, 50)
    if(type_graph == "linear" or type_graph == "quadratic"):
        plt.plot(xt, eval(str(thetas[0]/(-thetas[2])) +"+"+ str(thetas[1]/(-thetas[2])) +"*xt"))
    
    #plot the linear seperator boundary
    if(type_graph == "linear"):
        yt = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(xt, yt)
        z0 = np.empty(50*50)
        meshthetas = np.c_[X.reshape((50*50, 1)), Y.reshape((50*50, 1))]
        sigmainv = np.linalg.inv(sigma0)
        
        for point in range(50*50):
            x1 = meshthetas[point]
            z0[point] = np.exp(-1*(np.dot((x1 - mu0).dot(sigmainv), (x1 - mu0).T)))
            
        z0 = z0.reshape((50, 50))
        
        z1 = np.empty(50*50)
        for point in range(50*50):
            x1 = meshthetas[point]
            z1[point] = np.exp(-1*(x1 - mu1).dot(sigmainv).dot((x1 -mu1).T))
            
        z1 = z1.reshape((50, 50))
        
        plt.contour(xt, yt, z0)
        plt.contour(xt, yt, z1)
        plt.savefig("GDA_linear_boundary.jpg")
        
    #plot the quadratic decision boundary    
    elif(type_graph == "quadratic"):
        yt = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(xt, yt)
        z = np.empty(50*50)
        meshthetas = np.c_[X.reshape((50*50, 1)), Y.reshape((50*50, 1))]
        sigma0inv = np.linalg.inv(sigma0)
        sigma1inv = np.linalg.inv(sigma1)
        for point in range(50*50):
            x1 = meshthetas[point]
            z[point] = (x1-mu1).T.dot(sigma1inv).dot(x1-mu1) - (x1-mu0).T.dot(sigma0inv).dot(x1-mu0)
        z= z.reshape((50, 50))
        
        C = -np.log(np.linalg.det(sigma1)/np.linalg.det(sigma0))
        plt.contour(xt, yt, z, [C])
        plt.savefig("GDA_quadratic_boundary.jpg")
        
    plt.legend()
    plt.title('GDA')
    if(type_graph == "dummy"):
        plt.savefig("GDA_train_data.jpg")
    elif(type_graph == "linear"):
        plt.savefig("GDA_linear_boundary.jpg")
    elif(type_graph == "quadratic"):
        plt.savefig("GDA_quadratic_boundary.jpg")
    plt.show()
        
def main():
    #firstly read the data from the  file
    X_in = pd.read_csv("./lin_log/q4x.dat").values
    Y_in = pd.read_csv("./lin_log/q4y.dat").values
    #norlaize the data 
    X1 = modify_input(X_in)
    X = normalize(X1)
    print(X[0])
    #Evaluate the thetas and plot the decision boundary
    phi, mu0, mu1, sigma = compute_linear_parameters(X, Y_in)
    thetas = compute_linear_boundary(phi, mu0, mu1, sigma, X)
    print("mu0",mu0)
    print("mu1",mu1)
    print("sigma", sigma)
    print("phi", phi)
    print("thetas",thetas)
    
    #plot the scatter plot of train data and plot the linear boundary
    plot_graph(X, Y_in, phi, mu0, mu1, sigma, [], thetas,"dummy")
    plot_graph(X, Y_in, phi, mu0, mu1, sigma, [], thetas, "linear")
    
    #compute the parameters for quadratic boundary and plot it
    phi, mu0, mu1, sigma0, sigma1 = compute_quadratic_parameters(X, Y_in)
    print("mu0",mu0)
    print("mu1",mu1)
    print("sigma0", sigma0)
    print("sigma1", sigma1)
    print("phi", phi)
    plot_graph(X, Y_in, phi, mu0, mu1, sigma0, sigma1, thetas, "quadratic")
    
if __name__ == "__main__":
    main()
    