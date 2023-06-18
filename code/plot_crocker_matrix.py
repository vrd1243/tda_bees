#!/bin/python

import numpy as np
from matplotlib import pyplot as plt
import sys

prefix = sys.argv[1]

matrix = np.loadtxt('../results/{}_crocker_matrix.txt'.format(prefix), delimiter=',')
matrix = matrix[::20,::10]
x,y  = np.meshgrid(20*np.arange(0,matrix.shape[0],1), np.arange(0,matrix.shape[1],1))
	
plt.figure(dpi=300, figsize=(10,6))
levels=10   
plt.contour(x, y, matrix[::1,::1].T,linewidths=1, levels=levels, colors='k')
hm = plt.contourf(x, y, matrix[::1,::1].T, levels=levels, cmap = plt.cm.jet_r)
ax = plt.gca()  
ax.set_xlabel('Time(sec)')
ax.set_ylabel('Epsilon')
plt.savefig('../results/{}_crocker_plot.png'.format(prefix))
