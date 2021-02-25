import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import logging
import scipy.stats as st
import os
import statistics
import csv

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from random import seed
from random import random
from random import randint
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Parameters
random_seed = 2
seed(random_seed)
num_samples = 20000
# num_samples = len(x_y_df)
graph_sample_start = 10
graph_error_max = 0.6
graph_error_min = 0.35
train_size_percent = [0.005, 0.0065, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1 , 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
# train_size_percent = [0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.065, 0.08, 0.1 , 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8]
train_size_list = [p * num_samples for p in train_size_percent]
x_columns = list(range(0,10,1))
y_column = [10]


# First function
def first_binary_function(x, e):
  return int((bool(x[0] or x[1]) and (bool(x[2]) != bool(x[3]))) != bool(e))

# Generating data with the first function
num_dimension = 10
# num_dimension = 2
error_percent = 10
x_y_df = pd.DataFrame(columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','Result']) 
# x_y_df = pd.DataFrame(columns = ['x0','x1','Result']) 
for i in range(num_samples):
  x_values = []
  error_value = 0
  if (randint(0,99) < error_percent):
    error_value = 1
  for d in range(num_dimension):
    x_values.append(randint(0,1))
  x_values.append(first_binary_function(x_values, error_value))
  x_y_df.loc[i] = x_values


# Calculating errors
def error_calculation(y, y_hat):
  num_errors = 0
  num_sam = len(y)
  for i in range(num_sam):
    if(y[i] != y_hat[i]):
      num_errors += 1
  return(num_errors/num_sam)

# Plotting errors
def error_plotter(axes, title, train_size, train_error, test_error):
  x_flat = np.arange(graph_sample_start, num_samples, 100)
  axes.plot(x_flat,[0.1]*len(x_flat),  '-k', linestyle="-", linewidth = 4, color='silver')
  axes.scatter(train_size ,train_error) 
  axes.scatter(train_size ,test_error) 
  SMALL_SIZE = 12
  MEDIUM_SIZE = 12
  BIGGER_SIZE = 14
  axes.axis([100, num_samples, graph_error_min, graph_error_max])
  axes.set_xscale("log")
  axes.set_title(title)
  axes.set_xlabel("number of train samples")
  axes.set_ylabel("error")


# Neural Network
def my_nn(x_train, x_test, y_train, y_test, activation='relu', hidden_layer_sizes=(100,),max_iter=300,alpha=0.0001):
  clf =  MLPClassifier(activation=activation, random_state=random_seed, max_iter=max_iter, hidden_layer_sizes=hidden_layer_sizes,alpha=alpha)
  clf.fit(x_train, y_train)
  y_hat_train = clf.predict(x_train)
  y_hat_test = clf.predict(x_test)
  return(error_calculation(y_train, y_hat_train), error_calculation(y_test, y_hat_test))
  tree.plot_tree(clf)

activation_list = ['identity', 'logistic', 'tanh', 'relu']
# activation_list = ['relu']
# hidden_layer_sizes_list = [(3,), (10,), (30,), (100,), (300,) , (10, 10,), (10, 10, 10,)]
hidden_layer_sizes_list = [(10,)]
# max_iter_list=[10,30,100,300]
max_iter_list=[300]
# alpha_list = [0.0001, 0.001, 0.01, 0.1]
alpha_list = [0.0001]
num_cols = 2
subplot_size = len(activation_list)*len(hidden_layer_sizes_list)*len(max_iter_list)*len(alpha_list)
figure, axes = plt.subplots(nrows=int(np.ceil(subplot_size/num_cols)), ncols=num_cols, figsize=(12, 8), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
index = 0
for activation in activation_list:
  for hidden_layer_sizes in hidden_layer_sizes_list:
    for max_iter in max_iter_list:
      for alpha in alpha_list:
        train_error_list = []
        test_error_list = []
        for train_size_iterator in train_size_percent:
          x_train, x_test, y_train, y_test = train_test_split(x_y_df.iloc[:,x_columns], x_y_df.iloc[:,y_column], train_size=train_size_iterator, random_state=1)
          y_train=y_train.astype('int')
          y_test=y_test.astype('int')
          train_size = train_size_iterator * num_samples
          train_error, test_error =  my_nn(x_train, x_test, y_train.Result.tolist(), y_test.Result.tolist(),
                                           activation=activation, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha)
          train_error_list.append(train_error)
          test_error_list.append(test_error)
        title = "activation="+str(activation)
        # title = "hidden_layer_sizes="+str(hidden_layer_sizes)
        # title = "max_iter="+str(max_iter)
        # title = "regularizer factor="+str(alpha)
        error_plotter(axes[int(index/num_cols), int(index%num_cols)], title, train_size_list, train_error_list, test_error_list)
        index+=1


# Decision Tree
def my_dt(x_train, x_test, y_train, y_test, max_depth=3):
  clf = tree.DecisionTreeClassifier(max_depth=max_depth)
  clf.fit(x_train, y_train)
  y_hat_train = clf.predict(x_train)
  y_hat_test = clf.predict(x_test)
  return(error_calculation(y_train, y_hat_train), error_calculation(y_test, y_hat_test))
  tree.plot_tree(clf)  

depth_list = [1,2,3,4,5,6,7,8,9,10]
num_cols = 4
figure, axes = plt.subplots(nrows=int(np.ceil(len(depth_list)/num_cols)), ncols=num_cols, figsize=(15, 12), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})

for index, max_depth in enumerate(depth_list):
  train_error_list = []
  test_error_list = []
  for train_size_iterator in train_size_percent:
    x_train, x_test, y_train, y_test = train_test_split(x_y_df.iloc[:,x_columns], x_y_df.iloc[:,y_column], train_size=train_size_iterator, random_state=1)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    train_size = train_size_iterator * num_samples
    train_error, test_error =  my_dt(x_train, x_test, y_train.Result.tolist(), y_test.Result.tolist(), max_depth=max_depth)
    train_error_list.append(train_error)
    test_error_list.append(test_error)
  title = "max_depth="+str(max_depth)
  error_plotter(axes[int(index/num_cols), int(index%num_cols)], title, train_size_list, train_error_list, test_error_list)


# Boosting
def my_boosting(x_train, x_test, y_train, y_test, max_depth=3, n_estimators=100):
  clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=0, base_estimator=tree.DecisionTreeClassifier(max_depth=max_depth,random_state=random_seed))
  clf.fit(x_train, y_train)
  y_hat_train = clf.predict(x_train)
  y_hat_test = clf.predict(x_test)
  return(error_calculation(y_train, y_hat_train), error_calculation(y_test, y_hat_test))
  tree.plot_tree(clf)  

# depth_list = [1,2,3]
depth_list = [2]
n_estimators_list=[10,30,100,300]
# n_estimators_list=[100]
num_cols = 2
subplot_size = len(depth_list)*len(n_estimators_list)
figure, axes = plt.subplots(nrows=int(np.ceil(subplot_size/num_cols)), ncols=num_cols, figsize=(12, 8), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
index=0
for max_depth in depth_list:
  for n_estimators in n_estimators_list:
    train_error_list = []
    test_error_list = []
    for train_size_iterator in train_size_percent:
      x_train, x_test, y_train, y_test = train_test_split(x_y_df.iloc[:,x_columns], x_y_df.iloc[:,y_column], train_size=train_size_iterator, random_state=1)
      y_train=y_train.astype('int')
      y_test=y_test.astype('int')
      train_size = train_size_iterator * num_samples
      train_error, test_error =  my_boosting(x_train, x_test, y_train.Result.tolist(), y_test.Result.tolist(), max_depth=max_depth,n_estimators=n_estimators)
      train_error_list.append(train_error)
      test_error_list.append(test_error)
    # title = "max_depth="+str(max_depth)
    title = "n_estimators="+str(n_estimators)
    error_plotter(axes[int(index/num_cols), int(index%num_cols)], title, train_size_list, train_error_list, test_error_list)
    index+=1


# SVM
def my_svm(x_train, x_test, y_train, y_test, degree_param=1, kernel_param="linear", gamma_param = "scale", coef0_param=0):
  clf = svm.SVC(kernel=kernel_param)
  clf.fit(x_train, y_train)
  y_hat_train = clf.predict(x_train)
  y_hat_test = clf.predict(x_test)
  return(error_calculation(y_train, y_hat_train), error_calculation(y_test, y_hat_test))

kernel_param_list = ["linear", "poly", "rbf"]
num_cols = 2
subplot_size = len(kernel_param_list)
figure, axes = plt.subplots(nrows=int(np.ceil(subplot_size/num_cols)), ncols=num_cols, figsize=(12, 8), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
index=0
for kernel_param in kernel_param_list:
  train_error_list = []
  test_error_list = []
  for train_size_iterator in train_size_percent:
    x_train, x_test, y_train, y_test = train_test_split(x_y_df.iloc[:,x_columns], x_y_df.iloc[:,y_column], train_size=train_size_iterator, random_state=1)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    train_size = train_size_iterator * num_samples
    train_error, test_error =  my_svm(x_train, x_test, y_train.Result.tolist(), y_test.Result.tolist(), kernel_param=kernel_param)
    train_error_list.append(train_error)
    test_error_list.append(test_error)
  title = "kernel_param="+str(kernel_param)
  error_plotter(axes[int(index/num_cols), int(index%num_cols)], title, train_size_list, train_error_list, test_error_list)
  index+=1

# KNN
def my_knn(x_train, x_test, y_train, y_test, n_neighbors=3):
  neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
  neigh.fit(x_train, y_train)
  y_hat_test = neigh.predict(x_test).tolist()
  y_hat_train = neigh.predict(x_train).tolist()
  return(error_calculation(y_train, y_hat_train), error_calculation(y_test, y_hat_test))

n_neighbors_list = [1, 2, 3, 4, 6, 8, 10, 12]
num_cols = 4
subplot_size = len(n_neighbors_list)
figure, axes = plt.subplots(nrows=int(np.ceil(subplot_size/num_cols)), ncols=num_cols, figsize=(16, 8), gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
index=0
for n_neighbors in n_neighbors_list:
  train_error_list = []
  test_error_list = []
  for train_size_iterator in train_size_percent:
    x_train, x_test, y_train, y_test = train_test_split(x_y_df.iloc[:,x_columns], x_y_df.iloc[:,y_column], train_size=train_size_iterator, random_state=1)
    y_train=y_train.astype('int')
    y_test=y_test.astype('int')
    train_size = train_size_iterator * num_samples
    train_error, test_error =  my_knn(x_train, x_test, y_train.Result.tolist(), y_test.Result.tolist(), n_neighbors=n_neighbors)
    train_error_list.append(train_error)
    test_error_list.append(test_error)
  # print("k:", k_param) 
  # print("train size number of samples:", train_size_list)
  # print("train error:", train_error_list)
  # print("test error", test_error_list)
  title = "n_neighbors="+str(n_neighbors)
  error_plotter(axes[int(index/num_cols), int(index%num_cols)], title, train_size_list, train_error_list, test_error_list)
  index+=1



