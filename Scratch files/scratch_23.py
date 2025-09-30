import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select features for clustering
X = X[['sepal length (cm)', 'sepal width (cm)']]

# Choose an appropriate number of clusters k
k = 3

# Fit K-means algorithm to the data
kmeans = KMeans(n_clusters=k)
kmeans.fit(X)

# Print labels and centroids
print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)

# Scatter plot of the data with color indicating clusters
plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=300, c='red')
plt.title('K-means Clustering(K=3)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

# Experiment with different values of k
# In this case, since we know that there are three species of iris in the dataset,
# it's reasonable to experiment with k=2, k=3, and k=4 and compare the results.
k_values = [2, 3, 4, 5, 6]
inertia_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Plotting inertia values for different k values
plt.plot(k_values, inertia_values, marker='o')
plt.title('Inertia vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()

# Based on the elbow method, we choose the value of k where the inertia starts to decrease more slowly,
# which indicates diminishing returns in terms of adding more clusters.
# In this case, from the plot, k=3 seems to be the best choice.
