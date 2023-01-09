# K-Means Clustering

This page demostrates some sample code for k-means clustering. Two Python files has been shared.

One is written from scratch without any Machine Learning libraries such as `scikit-learn`.

Other uses the capabilities of `scikit-learn` tools.

## Elbow Method

`k_means_sklearn.py` examines the optimal k value by training the model and plotting the sum of squares for k from 1 to 9:

<img src="https://user-images.githubusercontent.com/22200109/211331365-14442f4a-62b7-4724-9dd4-4900bed55c62.png" width="500">

k is chosed as 3 where the elbow bends.

## Plotting the Clusters

The number of dimensions of data is reduced into 2D for visualization, with the help of `PCA` method.

After applying the clustering, the data is grouped as seen below:

<img src="https://user-images.githubusercontent.com/22200109/211332468-13313bad-fcd9-48c6-81a5-cde9b590de07.png" width="500">
