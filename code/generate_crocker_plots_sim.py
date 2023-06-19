#!/bin/python

import numpy as np
import pandas as pd
import cv2
import glob, os
import re
import collections

from PIL import Image
from skimage import data
from skimage.filters import threshold_otsu
from collections import defaultdict
from matplotlib.pyplot import cm
from skimage.filters import threshold_mean, threshold_otsu, threshold_minimum, threshold_local
import gudhi as gd
import networkx as nx
import re
from tqdm import tqdm

def make_crocker(loadpath, savepath):
    df = pd.read_csv(loadpath)

    m = re.search('\[Who x y \[(.+)\]\]', df.iloc[1,0])
    first = m.group(1)
    m = re.search('(\[.+\]) \[.+', first)

    all_comps = []
    all_avg_cluster_size = []
    all_max_cluster_size = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            m = re.search('\[Who x y \[(.+)\]\]', row[0])
            if m == None:
                print(row[0])
            first = m.group(1)
            result = np.array([float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", first)])
            point_cloud = result.reshape((-1,3))[:,1:]

            rips_complex = gd.RipsComplex(points=point_cloud).create_simplex_tree(max_dimension=1)
            pers = rips_complex.persistence()
            skeleton = rips_complex.get_skeleton(1)

            components = []
            avg_cluster_size = []
            max_cluster_size = []
            for epsilon in np.arange(0,20,1):
                edges = []
                G = nx.Graph()
                [G.add_node(n) for n in range(len(point_cloud))]

                for s in skeleton: 

                    nodes, filtration = s

                    if filtration <= epsilon and len(nodes) == 2:
                        G.add_edge(nodes[0], nodes[1])

                cc = nx.connected_components(G)

                clusters = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

                avg_cluster_size.append(np.mean(clusters))
                max_cluster_size.append(np.max(clusters))
                components.append(rips_complex.persistent_betti_numbers(epsilon,epsilon)[0])

            all_comps.append(np.array(components))
            all_avg_cluster_size.append(np.array(avg_cluster_size))
            all_max_cluster_size.append(np.array(max_cluster_size))
        except:
            print('pass idx: ', index)
            
    np.savetxt(savepath, all_comps, delimiter=',')

def main():
    
    if len(sys.argv) != 3:
        print("Usage: python generate_crocker_matrix.py <input_sim_file> <output_prefix>")
        exit(0)
        
    loadpath = sys.argv[1]
    output_prefix = sys.argv[2]
    savepath = '../results/{}_crocker_matrix.txt'.format(output_prefix)
    
    make_crocker(loadpath, savepath)

if __name__ == '__main__':
    main()
