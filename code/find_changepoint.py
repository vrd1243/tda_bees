#!/bin/python

import numpy as np
import sys

def find_overlap_len(labels):
    potential_changepoint = 0
    old_label = labels[0]
    found_change = False
    first_change = False
    first_change_time = 0
    
    for timestep, label in enumerate(labels):
        if label != old_label:
            potential_changepoint = timestep
            found_change = True
            if not first_change:
                first_change_time = timestep
            first_change = True
            #check to see if we hop back and forth 
            for next_label in labels[timestep + 1:]:
                if next_label == old_label:
                    found_change = False
                    break
            if found_change:
                return potential_changepoint - first_change_time
        old_label = label
    print('Failed to find suitable change point.')
    
def find_change(labels, first_changepoint = 0):
    #first_changepoint is for finding two changepoints. Set to the value of the first changepoint
    potential_changepoint = 0
    old_label = labels[first_changepoint]
    found_change = False
    
    for timestep, label in enumerate(labels[first_changepoint:]):
        if label != old_label:
            potential_changepoint = timestep
            found_change = True
            #check to see if we hop back and forth 
            for next_label in labels[timestep + 1:]:
                if next_label == old_label:
                    found_change = False
                    break
            if found_change:
                return potential_changepoint
        old_label = label
    print('Failed to find suitable change point.')
    
def main():

    if len(sys.argv) != 2:
        print("Usage: python find_changepoint.py <input_representation_file>")
        exit(0)
        
    loadpath = sys.argv[1]
    
    labels = np.loadtxt(loadpath, delimiter = ',')[:, 1]
    
    print('Changepoint:' , find_change(labels))
    
if __name__ == '__main__':
    main()