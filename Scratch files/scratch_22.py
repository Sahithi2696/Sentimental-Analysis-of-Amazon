import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame for the data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(df.head())

selected_features = ['sepal length (cm)', 'sepal width (cm)']
X = iris.data  # Using all features for clustering


# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using the Elbow Method to find the optimal number of clusters (k)
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):  # Trying k from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Inertia is another name for WCSS

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.title('Elbow Method to Determine Optimal k')
plt.grid(True)
plt.show()

# Based on the Elbow Method, let's choose k = 3
k = 3

# Fit K-means to the data with k clusters
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Get the cluster labels for each data point
labels = kmeans.labels_

# Add the cluster labels to the original Iris DataFrame
df['Cluster'] = labels

# Print the first few rows of the DataFrame with cluster labels
print(df.head())

# Get the cluster centers (centroids)
centroids = kmeans.cluster_centers_
print("\nCluster Centers:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1} Center:", centroid)
