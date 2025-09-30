import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame with the data
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Select the appropriate features for clustering
X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]

# Perform Agglomerative Clustering with different linkage criteria and distance metrics
linkage_methods = ['ward', 'complete', 'average', 'single']
distance_metrics = ['euclidean', 'cosine']

fig, axs = plt.subplots(len(linkage_methods), len(distance_metrics), figsize=(15, 12))

# Iterate through different linkage methods and distance metrics
for i, linkage_method in enumerate(linkage_methods):
    for j, distance_metric in enumerate(distance_metrics):
        if linkage_method == 'ward' and distance_metric != 'euclidean':
            continue  # Skip combinations that don't support 'ward' method

        # Create Agglomerative Clustering model
        agglomerative = AgglomerativeClustering(linkage=linkage_method, n_clusters=3)
        labels = agglomerative.fit_predict(X)

        # Compute the linkage matrix
        Z = linkage(X, method=linkage_method, metric=distance_metric)

        # Plot the dendrogram
        dendrogram(Z, ax=axs[i, j], labels=labels)
        axs[i, j].set_title(f'Linkage: {linkage_method.capitalize()}\nMetric: {distance_metric.capitalize()}')
        axs[i, j].set_xlabel('Samples')
        axs[i, j].set_ylabel('Distance')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
