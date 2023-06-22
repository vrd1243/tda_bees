import numpy as np
import os
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

#creates the norm and time tuples
def crocker_norm(crocker, time=True):
    
    norms = np.linalg.norm(crocker, axis=1)
    if not time:
        return norms
    return [[norm, t] for t, norm in enumerate(norms)]

#performs agglomerative clustering and saves results
def agg_clustering(data, num_clusters, savepath):
    agg = AgglomerativeClustering(n_clusters = num_clusters)
    agg.fit(data)
    #save tuple of (timestep, label)
    np.savetxt(savepath, [[t, label] for t, label in enumerate(agg.labels_)], delimiter=',')
    
#performs kmeans clustering and saves results
def kmeans_clustering(data, num_clusters, savepath):
    kmeans = KMeans(n_clusters = num_clusters, random_state = 0)
    kmeans.fit(data)
    np.savetxt(savepath, [[t, label] for t, label in enumerate(kmeans.labels_)], delimiter=',')

def main():
    
    # Load the PCA-reduced version of the CROCKER plot directly.    
    load_path = sys.argv[1]
    output_prefix = sys.argv[2]
    
    num_clusters = 2
    
    #first principal component and time tuples are saved in a txt file
    pca = np.loadtxt(load_path, delimiter=',', skiprows=1)

    cluster_labels_savepath = '../results/' + output_prefix
    
    #agglomerative clustering
    agg_clustering(pca, num_clusters, cluster_labels_savepath + '_agg_with_pca_time.txt')

    #kmeans clustering
    kmeans_clustering(pca, num_clusters, cluster_labels_savepath + '_kmeans_with_pca_time.txt')

if __name__ == '__main__':
    
    main()
