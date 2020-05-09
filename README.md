# K-Means


## Description

`find_k.py` finds the best guess for k clusters using the Elbow Method and the Sum of Square Distances

## To Run

Set the following values:

Parameter|Description|Line
---------|------------|----
data_path | path on local machine to txt data file| 14
min_k | lowest value of k to test| 18
max_k | highest value of k to test| 19
max_iter| number of interations to perform while optimizing centroids| 22

<br />

Invoke script using the the command:

`python3 find_k.py`

<br />
The function run_kmeans(data_path,min_k,max_k,max_iter) is invoked 5 times to ensure best estimate of k due to the Elbox Method. The most frequent result is returned as the final k value. The Elbow Method returns the best k value according to observance of at least a 7% drop in SSD for each k. This number was determined by trial and error. Another method to experiment with in the future is the Silhouette Method.  <br />

## Results

Final result will be printed to the console 

<br >
Created by: Breanna M Lee 
May 9, 2020
