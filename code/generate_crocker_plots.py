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

def desired_objects(bin_img):
    k= 115
    bee_size = 2900
    bee_locations=[]

    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=4)
    sizes = stats[:,-1]
    
    max_label = []
    min_size = 0
    comp_size = []

    for i in range(1, num_regions):
        if sizes[i] > min_size + 1000 :
            max_label.append(i)
            comp_size.append(sizes[i])

            num_bees = int(np.around(sizes[i]/bee_size)) 
            
            x_loc, y_loc = np.where(regions == i)

            for j in range(num_bees):
                i = np.random.randint(x_loc.shape[0])
                bee_locations.append([x_loc[i], y_loc[i]])
        else: 
            continue

    bee_locations = np.array(bee_locations)
    

    return bee_locations


# Run a simplicial complex on every frame and generate a 
# Betti_0 CROCKER matrix. 

def generate_crocker_matrix(next_img):

    all_comps = []
    all_avg_cluster_size = []
    all_max_cluster_size = []
    broken_frames=0

    G = nx.Graph()

    for i in tqdm(range(len(next_img))):
        arr = Image.open(next_img[i])
        data_gray = np.asarray(arr)

        kernel = np.ones((7,7), np.uint8)
        k = 5
        im_blur = cv2.GaussianBlur(src = data_gray, ksize = (k, k), sigmaX = 0)


        thresh_min = threshold_minimum(data_gray)
        binary_min = im_blur > thresh_min + 5
        binary_min_arr = np.array(binary_min, dtype=np.uint8)
        closing = cv2.morphologyEx(binary_min_arr, cv2.MORPH_CLOSE, kernel)
        data = 255 - closing * 255    

        point_cloud = desired_objects(data)

        rips_complex = gd.RipsComplex(points=point_cloud).create_simplex_tree(max_dimension=1)
        pers = rips_complex.persistence()
        skeleton = rips_complex.get_skeleton(1)
        components = []
        avg_cluster_size = []
        max_cluster_size = []

        for epsilon in np.arange(0,700,1):

            edges = []
            G = nx.Graph()
            [G.add_node(n) for n in range(len(point_cloud))]

            for s in skeleton: 
                
                nodes, filtration = s
                e = 0
                
                if filtration <= epsilon and len(nodes) == 2 and e < 60:
                    e = e + 1
                    G.add_edge(nodes[0], nodes[1])
                else: 
                    break

            cc = nx.connected_components(G)
            clusters = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

            avg_cluster_size.append(np.mean(clusters))
            max_cluster_size.append(np.max(clusters))
            components.append(rips_complex.persistent_betti_numbers(epsilon,epsilon)[0])

        if (np.array(components[0])> 65) or (np.array(components[0]) < 35) :    
            print ('Noisy frame #{} -> removed'.format(broken_frames+1))
            broken_frames+=1
            components.pop()

        else:

            all_comps.append(np.array(components))
            all_avg_cluster_size.append(np.array(avg_cluster_size))
            all_max_cluster_size.append(np.array(max_cluster_size))

    #some frames were removed from the set because I have my hand in the setup, etc.
    print ('Total number of removed frames = ', broken_frames)

    return all_comps
    
def main():
    
    if len(sys.argv) != 3:
        print("Usage: python generate_crocker_matrix.py <input_frame_dir> <output_prefix>")
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
