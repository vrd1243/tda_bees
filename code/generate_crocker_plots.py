import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob, os
import re
import collections
import gudhi as gd
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['figure.figsize'] = (10,6)
import networkx as nx
import sys

from PIL import Image
from skimage import data
from skimage.filters import threshold_otsu
from collections import defaultdict
from matplotlib.pyplot import cm
from skimage.filters import threshold_mean, threshold_otsu, threshold_minimum, threshold_local
from tqdm import tqdm
from datetime import datetime
from scipy.stats import gaussian_kde

numbers = re.compile(r'(\d+)')

# Sorting of frames based on their time-stamp. 

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Extracting the approximate location of bees from the raw image. 
# This will serve as the point cloud data on which we run TDA 
# and generate CROCKER plots. 

def desired_objects(bin_img, seed):

    bee_size = 2000
    bee_locations=[]

    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=4)
    sizes = stats[:,-1]
    max_label = []
    min_size = 0
    comp_size = []
    np.random.seed(seed)

    for i in range(1, num_regions):
        if sizes[i] > min_size:
            max_label.append(i)
            comp_size.append(sizes[i])

        num_bees = int(np.around(sizes[i]/bee_size))    
        x_loc, y_loc = np.where(regions == i)

        for j in range(num_bees):
            i = np.random.randint(x_loc.shape[0])
            bee_locations.append([x_loc[i], y_loc[i]])
        
    bee_locations = np.array(bee_locations)
       
    return bee_locations


# Run a simplicial complex on every frame and generate a 
# Betti_0 CROCKER matrix. 

def generate_crocker_matrix(next_img):

    summarized_comps = np.zeros((len(next_img),700))
    summarized_avg_cluster_size = np.zeros((len(next_img),700))
    summarized_max_cluster_size = np.zeros((len(next_img),700))

    iterations = 1
    for r in range(iterations):

        G = nx.Graph()
        all_comps = []

        for i in tqdm(range(len(next_img))):

            arr = Image.open(next_img[i])
            data = np.asarray(arr)
            data_gray = np.average(data, axis=2)

            # Blur image to remove any graininess. 
            kernel = np.ones((5,5), np.uint8)
            k = 35
            im_blur = cv2.GaussianBlur(src = data_gray, ksize = (k, k), sigmaX = 0)

            # Threshold the image and convert to black and white. 
            thresh_min = threshold_minimum(im_blur)
            binary_min = im_blur > thresh_min
            binary_min_arr = np.array(binary_min, dtype=np.uint8)

            # Morphology: Dilate and erode to fill any isolated pixels that are not 
            # really bees. 
            closing = cv2.morphologyEx(binary_min_arr, cv2.MORPH_CLOSE, kernel)
            data = 255 - closing * 255    

            # Generate a point cloud data and get the rips complex. 
            point_cloud = desired_objects(data, r)
            rips_complex = gd.RipsComplex(points=point_cloud).create_simplex_tree(max_dimension=1)
            pers = rips_complex.persistence()
            skeleton = rips_complex.get_skeleton(1)

            components = []
            avg_cluster_size = []
            max_cluster_size = []

            # Iterate through the complex and get Betti counts for pre-determined epsilons.
            start_time = datetime.now()
            for epsilon in np.arange(0,700,1):

                edges = []
                G = nx.Graph()
                [G.add_node(n) for n in range(len(point_cloud))]

                for s in skeleton: 
                    nodes, filtration = s

                    if filtration <= epsilon and len(nodes) == 2:
                        G.add_edge(nodes[0], nodes[1])

                cc = nx.connected_components(G)
                clusters = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
                components.append(rips_complex.persistent_betti_numbers(epsilon,epsilon)[0])

            all_comps.append(np.array(components))
	
        summarized_comps += np.array(all_comps)

    summarized_comps = summarized_comps / iterations

    for i in np.where(summarized_comps[:,1] > 90)[0]:
        summarized_comps[i,:] = summarized_comps[i-1,:]

    return summarized_comps
    
def main():
    
    if len(sys.argv) != 3:
        print("Usage: python crocker_matrix.py <input_frame_dir> <output_prefix>")
        exit(0)
        
    map_dir = sys.argv[1]
    output_prefix = sys.argv[2]
    
    next_img = sorted(glob.glob(os.path.join(map_dir,'*.png' )), key=numericalSort)[::]
    summarized_comps = generate_crocker_matrix(next_img)
    np.savetxt('../results/{}_crocker_matrix.txt'.format(output_prefix), summarized_comps, delimiter=',')

    matrix = np.array(summarized_comps)
    matrix = np.array(matrix).astype(float)
    
    matrix = matrix[::20,::10]
    x,y  = np.meshgrid(20*np.arange(0,matrix.shape[0],1), np.arange(0,matrix.shape[1],1))

    plt.figure(dpi=300, figsize=(10,6))
    levels=10
    plt.contour(x, y, matrix[::1,::1].T,linewidths=1, levels=levels, colors='k')
    hm = plt.contourf(x, y, matrix[::1,::1].T, levels=levels, cmap = plt.cm.jet_r)
    ax = plt.gca()
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('Epsilon')
    plt.savefig('../results/{}_crocker_plot.png'.format(output_prefix))
    
if __name__ == '__main__':
    
    main()
