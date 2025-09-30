import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()

# Create a Pandas DataFrame for the data
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Assign the appropriate features to X
# For this example, let's use only the first two features: 'sepal length (cm)' and 'sepal width (cm)'
# You can adjust this based on the features you want to include
selected_features = ['sepal length (cm)', 'sepal width (cm)']
X = df[selected_features]

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a range of k values to experiment with
k_values = [2, 3, 4, 5]  # We will plot for these values of k

plt.figure(figsize=(14, 10))

for i, k in enumerate(k_values, start=1):
    # Fit K-means to the data with k clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)

    # Get the cluster labels for each data point
    labels = kmeans.labels_

    # Add the cluster labels to the original Iris DataFrame
    df['Cluster'] = labels

    # Get the cluster centers (centroids)
    centroids = kmeans.cluster_centers_

    # Plotting the data with colors indicating the clusters for each value of k
    plt.subplot(2, 2, i)
    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['sepal length (cm)'], cluster_data['sepal width (cm)'],
                    label=f'Cluster {cluster + 1}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title(f'K-means Clustering (k={k})',color='purple')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
