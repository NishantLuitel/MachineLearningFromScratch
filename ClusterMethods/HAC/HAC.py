import numpy as np
from copy import deepcopy

class HAC:
    def __init__(self, data):        
        
        self.data = data
        self.linkage = linkage
        self.distance_matrix = np.zeros((len(data), len(data)))
        self.clusters = [[i] for i in range(len(data))]
        self.cluster_history = []
        self.cluster_history.append(deepcopy(self.clusters))

    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2)**2))

    def update_distance_matrix(self, i, j):
        for point1 in self.clusters[i]:
            for point2 in self.clusters[j]:
                distance = self.euclidean_distance(self.data[point1], self.data[point2])
#                 if self.linkage == 'single':
#                     self.distance_matrix[point1][point2] = self.distance_matrix[point2][point1] = min(self.distance_matrix[point1][point2], distance)
#                 elif self.linkage == 'complete':
#                     self.distance_matrix[point1][point2] = self.distance_matrix[point2][point1] = max(self.distance_matrix[point1][point2], distance)
#                 elif self.linkage == 'average':
#                     self.distance_matrix[point1][point2] = self.distance_matrix[point2][point1] = (self.distance_matrix[point1][point2] * self.counts[point1][point2] +
#                                                                                                    distance) / (self.counts[point1][point2] + 1)
#                     self.counts[point1][point2] += 1

    def fit(self,linkage = 'single'):
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                self.distance_matrix[i][j] = self.distance_matrix[j][i] = self.euclidean_distance(self.data[i], self.data[j])
        self.counts = np.zeros((len(self.data), len(self.data)))
        while len(self.clusters) > 1:
            closest_clusters = (-1, -1, float('inf'))
            for i, cluster1 in enumerate(self.clusters):
                for j, cluster2 in enumerate(self.clusters[i + 1:]):
                    j = j + i + 1
                    
#                     average_distance = self.distance_matrix[cluster1[0]][cluster2[0]]
                    
                    if linkage == 'single':
                        average_distance = min([self.distance_matrix[m][n] for m in cluster1 for n in cluster2])
                    elif linkage == 'complete':
                        average_distance = max([self.distance_matrix[m][n] for m in cluster1 for n in cluster2])
                    elif linkage == 'average':
                        average_distance = sum([self.distance_matrix[m][n] for m in cluster1 for n in cluster2])/(len(cluster1)*len(cluster2))
                    
                    if average_distance < closest_clusters[2]:
                        closest_clusters = (i, j, average_distance)
            i, j, _ = closest_clusters
            self.clusters[i] = self.clusters[i] + self.clusters[j]
            self.cluster_history.append(deepcopy(self.clusters))
            del self.clusters[j]
            
#             print(i,j,len(self.clusters))
            self.update_distance_matrix(i, j-1)
    
    def get_clusters(self,num):
        assert num<=len(data) and num>1,"Number of clusters should be >1 and <= number of data points"
        return self.cluster_history[-(num-1)]