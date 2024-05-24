import numpy as np

class KMeans:
    def __init__(self, data, k, init_centroids=None, max_iters=100):
        self.data = data
        self.k = k
        self.init_centroids = init_centroids
        self.max_iters = max_iters
    
    @staticmethod
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def initialize_centroids(self):
        if self.init_centroids is not None:
            return self.init_centroids
        indices = np.random.choice(self.data.shape[0], self.k, replace=False)
        return self.data[indices]

    def assign_clusters(self, centroids):
        clusters = []
        for point in self.data:
            distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def update_centroids(self, clusters):
        new_centroids = []
        for i in range(self.k):
            cluster_points = self.data[clusters == i]
            new_centroids.append(cluster_points.mean(axis=0))
        return np.array(new_centroids)

    def calculate_average_distances(self, centroids, clusters):
        average_distances = []
        for i in range(self.k):
            cluster_points = self.data[clusters == i]
            if len(cluster_points) == 0:
                average_distances.append(0)
                continue
            distances = [self.euclidean_distance(point, centroids[i]) for point in cluster_points]
            average_distance = np.mean(distances)
            average_distances.append(average_distance)
        return np.array(average_distances)

    def kmeans(self):
        centroids = self.initialize_centroids()
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(centroids)
            new_centroids = self.update_centroids(clusters)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        average_distances = self.calculate_average_distances(centroids, clusters)
        return centroids, clusters, average_distances
