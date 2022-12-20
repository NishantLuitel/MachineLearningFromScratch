import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self,X,k=5,max_iter=50,):
        self.K = k
        self.max_iter = max_iter
        self.X = X
        
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        
    def cluster(self):
        self.n_samples, self.n_features = self.X.shape

        #initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace = False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        #optimization
        for _ in range(self.max_iter):
            #update clusters
            self.clusters = self._create_clusters(self.centroids)

            #update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            #check if converged
            if self._is_converged(centroids_old, self.centroids):
                break
                
            return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
            
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self._euclidean_distance(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _euclidean_distance(self,sample,point):
        return np.linalg.norm(sample-point)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroids_old, centroids):
        distances = [self._euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    
    def plot_scores_2d(self, colors, grid = True, dim_1 = 1, dim_2 = 2):
        data = self.X
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure(figsize=(15,10))
        plt.axhline(c = 'black', alpha = 0.2)
        plt.axvline(c = 'black', alpha = 0.2)
        plt.scatter(data.T[dim_1 - 1,:], data.T[dim_2 - 1,:], c = colors)
        plt.grid(grid)
        plt.title('Kmeans Clustering')
        plt.xlabel('{}st dimension'.format(dim_1))
        plt.ylabel('{}nd dimension'.format(dim_2))
        plt.show()