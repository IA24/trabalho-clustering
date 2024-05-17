import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans


num_samples = 50
max_limit = 3.5
min_limit = 0

def generate_coordinate():
    return np.random.uniform(min_limit, max_limit, num_samples)

descriptive = {'Feature 1': generate_coordinate(), 'Feature 2': generate_coordinate()}
dataset_features = pd.DataFrame(descriptive)

# Visualização dos dados
def setup_data():
    plt.scatter(dataset_features['Feature 1'].values, dataset_features['Feature 2'].values, s=100, c='grey', label='Data')
    plt.title('Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

setup_data()

# Inicialização dos centróides
centroids = np.array([[3,3.5], [3.5,3]])
indexes = ['C1', 'C2']
column_names = ['Feature 1', 'Feature 2'] 
dataset_centroids = pd.DataFrame(centroids,index = indexes, columns = column_names)

# Visualização dos dados com centróides
def setup_data_with_centroids():
    plt.scatter(dataset_features['Feature 1'].values, dataset_features['Feature 2'].values, s=100, c='grey', label='Data')
    plt.scatter (dataset_centroids ['Feature 1'].values, dataset_centroids['Feature 2'].values, s=250, marker = 'x', c = 'black')
    plt.title('Dataset and Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

setup_data_with_centroids()

kmeans = KMeans(dataset_features.values, 2, centroids)
centroids, clusters = kmeans.kmeans()

# Adicionando os clusters ao dataset
dataset_features['Cluster'] = clusters

# Separando os grupos para visualização
group_1 = dataset_features.loc[dataset_features['Cluster'] == 0]
group_2 = dataset_features.loc[dataset_features['Cluster'] == 1]

# Visualização dos clusters
def setup_clusters():
    plt.scatter(group_1['Feature 1'].values, group_1['Feature 2'].values, s=100, c='b', label='Group 1')
    plt.scatter(group_2['Feature 1'].values, group_2['Feature 2'].values, s=100, c='r', label='Group 2')
    plt.scatter(centroids[:,0], centroids[:,1], s=250, marker = 'x', c='black')
    plt.legend()
    plt.title('Clustered Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

setup_clusters()

# Resultados
def results():
    print('K-Means labels\n', clusters)
    print('K-Means cluster centers\n', centroids)

results()