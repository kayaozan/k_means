# This is a sample code of K-Means clustering via sklearn, written by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is named as Sales_Transactions_Dataset_Weekly Data Set.
# It contains weekly purchased quantities of 800 over products over 52 weeks.
# It can be obtained at https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

# Dataset provides normalized data.
x = data.filter(regex='Normalized')

# The number of dimensions is reduced into 2D for visualization.
x = PCA(2).fit_transform(x)

# K-Means is initialized and trained.
kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto")
ids = kmeans.fit_predict(x)

# Labels and cluster centers are obtained.
u_ids = np.unique(ids)
centroids = kmeans.cluster_centers_

#Plotting the data with their respected clusters and cluster centers
for i in u_ids:
    plt.scatter(x[ids == i , 0] , x[ids == i , 1] , label = i)

plt.scatter(centroids[:,0] , centroids[:,1] , s = 100, color = 'k')
plt.legend()
plt.show()