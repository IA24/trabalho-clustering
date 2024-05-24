import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from kmeans import KMeans  # Certifique-se de que o módulo KMeans está correto
from client import Client  # Certifique-se de que o módulo Client está correto
from constants import NUM_SAMPLES, X_LINE, Y_LINE, CLUSTER, POINT_SIZE

def create_clients(num_samples):
    return [Client() for _ in range(num_samples)]

def create_dataset_features(clients):
    descriptive = {X_LINE: [client.age for client in clients],
                   Y_LINE: [client.height for client in clients]}
    return pd.DataFrame(descriptive)

def plot_data(features, title='Dataset', centroids=None, clusters=None):
    plt.scatter(features[X_LINE].values, features[Y_LINE].values, s=100, c='grey', label='Data')
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=250, marker='x', c='black', label='Centroids')
    
    if clusters is not None:
        unique_clusters = np.unique(clusters)
        for cluster_num in unique_clusters:
            cluster_data = features[features[CLUSTER] == cluster_num]
            x_values = cluster_data[X_LINE].values
            y_values = cluster_data[Y_LINE].values
            random_color = np.random.rand(3,)
            hex_color = mcolors.to_hex(random_color)
            plt.scatter(x_values, y_values, s=POINT_SIZE, c=hex_color, label=f'Group {cluster_num + 1}')
    
    plt.title(title)
    plt.xlabel(X_LINE)
    plt.ylabel(Y_LINE)
    plt.legend()
    plt.show()

def main():
    clients = create_clients(NUM_SAMPLES)
    dataset_features = create_dataset_features(clients)
    plot_data(dataset_features)
    initial_centroids = np.array([[3, 3.5], [3.5, 3]])
    plot_data(dataset_features, title='Dataset and Centroids', centroids=initial_centroids)
    kmeans = KMeans(dataset_features.values, k=10)
    centroids, clusters, average_distances = kmeans.kmeans()
    dataset_features[CLUSTER] = clusters
    plot_data(dataset_features, title='Clustered Data', centroids=centroids, clusters=clusters)
    print('K-Means average distance\n', average_distances)

if __name__ == "__main__":
    main()
