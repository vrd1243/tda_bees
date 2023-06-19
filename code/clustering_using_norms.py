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
    
    if len(sys.argv) < 3:
        print("Usage: python clustering_using_norms.py <input_crocker_file> <output_prefix>") 
        exit(0)

    load_file = sys.argv[1]
    output_prefix = sys.argv[2]
    
    num_clusters = 2
    
    #use only the first 900 timesteps
    crocker = np.loadtxt(load_file, delimiter=',')[:900]
    
    #create Betti norms with and without time
    betti_norms_with_time = crocker_norm(crocker)
    betti_norms = crocker_norm(crocker, time=False).reshape(-1,1)
    
    load_path = os.path.dirname(load_file)
    
    if load_path == '':
        load_path = '.'
        
    cluster_labels_savepath = load_path + '/' + output_prefix

    np.savetxt(cluster_labels_savepath + '_norms.txt', betti_norms)
    
    #agglomerative clustering
    agg_clustering(betti_norms_with_time, num_clusters, cluster_labels_savepath + '_agg_with_time.txt')
    agg_clustering(betti_norms, num_clusters, cluster_labels_savepath + '_agg.txt')

    #kmeans clustering
    kmeans_clustering(betti_norms_with_time, num_clusters, cluster_labels_savepath + '_kmeans_with_time.txt')
    kmeans_clustering(betti_norms, num_clusters, cluster_labels_savepath + '_kmeans.txt')

if __name__ == '__main__':
    
    main()
