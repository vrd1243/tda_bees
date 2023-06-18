import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
#mpl.rcParams['figure.figsize'] = (10,6)
import networkx as nx
from datetime import datetime

numbers = re.compile(r'(\d+)')

# Sorting of 
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def desired_objects(bin_img, seed):
    #comp_dict = defaultdict(list)
    k= 115
    bee_size = 2000
    bee_locations=[]

    num_regions, regions, stats, centroids = cv2.connectedComponentsWithStats(bin_img.astype(np.uint8), connectivity=4)
    sizes = stats[:,-1]#np.argsort(-stats[:,-1]) 
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