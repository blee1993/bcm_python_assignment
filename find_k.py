#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:09:49 2020

@author: breannamarielee
"""

import numpy as np

def main():
    
     # Set path to data txt file
    data_path = '/Users/breannamarielee/Documents/bcm_python_assignment/datasets/0.txt' #ask if can import sys to make it a command-line input
    
    #############################  PARAMETERS ########################################
    # Set k values to test
    min_k = 2
    max_k = 8
    
    # Max Interations for Determining Centroids (used for Stopping Criteria)
    max_iter = 10
    
    ##################################################################################
    
    results = []
    for i in range(5): 
        print("##### RUN # " + str(i+1) + "#########")
        results.append(run_kmeans(data_path,min_k,max_k,max_iter))

    results_array = np.array(results)
    counts = np.bincount(results_array)     
    print("-----------------------------")
    print("Best K Value = " + str(np.argmax(counts)))
    print("-----------------------------")

def run_kmeans(data_path,min_k,max_k,max_iter):
    '''Determine optimal k clusters using Elbow Method
    
    input: data_path is the path to data txt file
           min_k is the min k clusters to test
           max_k is the max k clusters to test
           max_iter is the number of iterations to determine new centroids
            
    output: k-value selected by Elbow Method

    
    ''' 
    
    # Import data
    data = []
    count = 0
    for datapoint in open(data_path):
        datapoint_split_strings = datapoint.split()
        datapoint_split_floats = [float(i) for i in datapoint_split_strings]
        data.append(datapoint_split_floats)
        count += 1
        
        
    
    k_values = []
    SSD_values = []
    
    for k in range(min_k,max_k+1):
        print("Calulating SSD for k = " + str(k))
        k_values.append(k)    
        iteration = 0
        
        # Determine initial centroids at random from the dataset
        centroid_indices = []
        while len(centroid_indices) != k:
            random_index = np.random.randint(0,count) 
            if random_index not in centroid_indices:
                centroid_indices.append(int(random_index))
        
        centroids = [] # contains the corresponding coordinates of the randomly-selected centroids
        for i in range(len(centroid_indices)):
            centroids.append(data[centroid_indices[i]])
        
        while iteration <= max_iter:
            #print("ITERATION #: " + str(iteration))
            
            # Assign datapoints to an initial cluster(centroid)
            datapoint_centroid_assignment_matrix = []
            
            for i in range(len(data)):
                datapoint_index = i
                centroid_assignment = assign_cluster(data[i],centroids)
                datapoint_centroid_assignment_matrix.append([datapoint_index,centroid_assignment])
            
            # Update centroid
            centroids = update_centroids(datapoint_centroid_assignment_matrix,data,k)
            iteration += 1
            
    
        # Determine Sum of Squared Distance for this k
        SSD = sum_of_squared_distances(datapoint_centroid_assignment_matrix,centroids,data)
        SSD_values.append(SSD)
        
    return(best_k(k_values,SSD_values))
    
        
def l2_distance(a,b):
    '''Calculate the Euclidean Distance between 2 datapoints
    
    input: a is a list of one n-dimensional datapoint
           b is a list of one n-dimensional datapoint
           
           a and b should be of equal length
           
    output: dist is a float valued at the distance between a and b
    '''

            
    dist = 0
    for i in range(len(a)):
        dist = dist + (a[i]-b[i])**2
        
    return(dist)
    
   
def assign_cluster(a,centroids):
    '''Determine which cluster to assign to the given datapoint
    
    input: a is one n-dimensional datapoint
           centroids is a k-dimensional list containing the datapoints of cluster centroids 
    
    output: winning_cluster is the cluster number of the closed centroid (centroid_indices[winning_centroid_index] will provide the original dataset index nubmer if first iteration)
    
    '''
    
    winning_centroid_distance = l2_distance(a,centroids[0])   #initialize; distance between a and the first centroid
    winning_cluster = 0
    
    
    for i in range(len(centroids)):
        distance = l2_distance(a,centroids[i]) 
        if  distance < winning_centroid_distance:
            winning_centroid_distance = distance
            winning_cluster = i
    
    return(winning_cluster)
    
def update_centroids(datapoint_centroid_assignment_matrix,data,k_clusters):
    '''Returns the datapoints of k new centroids
    input: datapoint_centroid_assignment_matrix is a 2D list containing the index and assigned cluster of a given datapoint
           data is a 2D list containg values of all datapoints at each index
           k_clusters is the number of clusters
    output: new_centroids is a list containing the datapoints of the k new centroids based upon averaging datapoints of the previous cluster
    
    '''
        
    new_centroid = [[]]*k_clusters
    
    for datapoint in datapoint_centroid_assignment_matrix:
        # datapoint[1] = assigned cluster number
        # datapoint[0] = index of datapoint

        if not new_centroid[datapoint[1]]:
            new_centroid[datapoint[1]] = data[datapoint[0]]
        else:
            new_centroid[datapoint[1]] = [sum(x) for x in zip(new_centroid[datapoint[1]], data[datapoint[0]])] #add up features from each datapoint for each cluster
     
    for i in range(k_clusters):
        new_centroid[i] = np.divide(new_centroid[i],len(data))
        if new_centroid[i].size == 0:
            new_centroid[i] = [0]*len(data[0])
            
  
    
    return(new_centroid)

def sum_of_squared_distances(datapoint_centroid_assignment_matrix,centroids,data):
    '''Calculated the Sum of Squared Distance between the datapoints and their respective cluster centroid
    
    input: datapoint_centroid_assignment_matrix is a 2D list containing the index and assigned cluster of a given datapoint
           centroids is a list containing the datapoints of each cluster centroid
           data is a 2D list containg values of all datapoints at each index
        
    output: np.sum(cluster_sums) is a float valued at the SSD for the given datapoints and their respective clusters
        
        
 
    '''
    k_clusters = len(centroids)
    cluster_sums = [[0]]*k_clusters
    
    for datapoint in datapoint_centroid_assignment_matrix:
        # datapoint[1] = assigned cluster number
        # datapoint[0] = index of datapoint

        total = 0

        for j in range(len(data[datapoint[0]])): # j represents a feature, 0-999
            total = total + (data[datapoint[0]][j] - centroids[datapoint[1]][j])**2
            
            cluster_sums[datapoint[1]] = cluster_sums[datapoint[1]] + total
            
    return(np.sum(cluster_sums))

def best_k(k_values,SSD_values):
    '''Determine the best k using Elbow-Method and SSD values
    
    input: k_values is a list of k-values, or number of clusters
           SSD_values is a list containing the repective sum of squared distance values for each k
        
    output: final_k is the best k-value determined by minimal changes between the current SSD and the next k's SSD
        
    '''
    if len(k_values)==1:
        return (k_values[0])
    else:
        tol = 7 #percent drop needed to select a cluster
        SSD_values = [SSD/10e8 for SSD in SSD_values]
        SSD_comparator = SSD_values[0]
        percent_changes = []
    
        for i in range(1,len(k_values)):
            percent_change = np.divide( (SSD_values[i] - SSD_comparator), np.abs(SSD_comparator) ) * 100
            percent_changes.append(percent_change)     
            SSD_comparator = SSD_values[i]

        final_k = k_values[len(k_values)-1] # initiialize
         
        for i in range(0,len(percent_changes)):
            if (percent_changes[i]) <= -(tol): 
                final_k = k_values[i+1]
                return(final_k)
  
    return(final_k)
    

if __name__ == "__main__":
    main()
    
   