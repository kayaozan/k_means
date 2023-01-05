# This is a sample code of K-Means clustering written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is named as Sales_Transactions_Dataset_Weekly Data Set.
# It contains weekly purchased quantities of 800 over products over 52 weeks.
# It can be obtained at https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly

def initialize_means(x, k):
# Mean initialization based on the k-means++ algorithm.

    # First cluster centroid is selected randomly.
    index = np.random.randint(x.shape[0])
    means = x[index,:].reshape(1,-1)

    # Remaining centroids are to be calculated in this loop.
    for _ in range(1,k):
        maxDist = 0

        for j in range(x.shape[0]):
            
            # The distance to the closest centroid for each data sample is calculated.
            Dist = np.min(np.sum((means - x[j,:])**2, axis = 1))

            # The largest distance and the corresponding index are stored
            if Dist > maxDist:
                maxDist = Dist
                index = j
            
        # The data point of the largest distance is saved as the next centroid.
        means = np.vstack((means, x[index,:].reshape(1,-1)))

    return means

def closest_mean(x, means):
# Checks the closest point of mean for each sample and saves it.
# The index of a sample is also the index of its ID.

    idx = []
    for i in range(x.shape[0]):
        idx.append( np.argmin(np.sum((means - x[i,:])**2, axis = 1)) )

    return np.array(idx)

def calculate_means(x, idx, means):
# Calculates the new means for each ID of clusters.

    for i in range(means.shape[0]):
        means[i,:] = np.sum(x[idx == i,:], axis = 0) / np.sum(idx == i)

    return means

def error_function(x, idx, means):
# Calculates the errors of each sample with their selected means.

    E = 0
    for i in range(x.shape[0]):
        E = E + np.sum((x[i,:] - means[idx[i],:])**2)

    return (1 / x.shape[0]) * E

def k_means(x, k = 8, max_iter=100, tol=0.00001):
# K-Means function.
# For the chance of finding a local optima, it gets run for several times (trials).

    means = initialize_means(x, k)
    E = 100_000_000 # An arbitrary big number

    for j in range(max_iter):
        idx = closest_mean(x, means)
        means = calculate_means(x, idx, means)
        E_ = error_function(x, idx, means)

        # If the update of error is not significant, loop gets stopped.
        if E - E_ < tol:
            break
        else:
            E = E_

    return idx, means

import pandas as pd
import numpy as np

data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
x = data.to_numpy()[:,1:53].astype(dtype = 'float')

# Data is sent to the model to get the cluster ids and cluster centroids.
idx, means = k_means(x)