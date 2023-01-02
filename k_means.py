# This is a sample code of K-Means clustering written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is named as Sales_Transactions_Dataset_Weekly Data Set.
# It contains weekly purchased quantities of 800 over products over 52 weeks.
# It can be obtained at https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly

def initialize_means(x, k):
# Means are initialized by randomly selecting k samples from data.

    indices = np.random.choice(x.shape[0], size = k, replace = False)
    means = x[indices,:]

    return means

def closest_mean(x, means):
# Checks the closest point of mean for each sample and saves it.
# The index of a sample is also the index of its ID.

    idx = []
    for i in range(x.shape[0]):
        id = np.argmin(np.sum((means - x[i,:])**2, axis = 1))
        idx.append(id)

    return idx

def calculate_means(x, idx, means):
# Calculates the new means for each ID of clusters.

    for i in range(means.shape[0]):
        means[i,:] = np.sum(x[np.array(idx) == i,:], axis = 0) / np.sum(np.array(idx) == i)

    return means

def error_function(x, idx, means):
# Calculates the errors of each sample with their selected means.

    E = 0
    for i in range(x.shape[0]):
        E = E + np.sum((x[i,:] - means[idx[i],:])**2)

    return (1 / x.shape[0]) * E

def k_means(x, k = 8, trials = 20, max_iter=100, tol=0.000001):
# K-Means function.
# For the chance of finding a local optima, it gets run for several times (trials).

    errors = []
    IDX = []
    for i in range(trials):
        means = initialize_means(x, k)
        idx = closest_mean(x, means)
        E = error_function(x, idx, means)

        for j in range(max_iter):
            idx = closest_mean(x, means)
            means = calculate_means(x, idx, means)
            E_update = error_function(x, idx, means)

            # If the update of error is not significant, loop gets stopped.
            if (E - E_update) < tol:
                break
            else:
                E = E_update

        # Error and IDs are stored to pick ones with the lowest error later.
        errors.append(E_update)
        IDX.append(idx)

    return IDX[np.argmin(errors)]

import pandas as pd
import numpy as np

data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

# Dataset provides normalized data.
x = data.filter(regex='Normalized').to_numpy()

# Input data is sent to the model to get the cluster ids.
idx = k_means(x)