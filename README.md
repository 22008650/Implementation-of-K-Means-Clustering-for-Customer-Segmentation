# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages
2. Insert the dataset to perform the k - means clustering
3. perform k - means clustering on the dataset
4. Then print the centroids and labels
5. Plot graph and display the clusters

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MALARVIZHI G
RegisterNumber:  212222040096
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers_EX8.csv")
data
X=data[["Annual Income (k$)","Spending Score (1-100)"]]
X
plt.figure(figsize=(4,4))
plt.scatter(data["Annual Income (k$)"],data["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
k =5
kmeans=KMeans(n_clusters=k)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels=kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors=['r','g','b','c','m']
for i in range(k):
  cluster_points = X[labels==i]
  plt.scatter(cluster_points["Annual Income (k$)"],cluster_points["Spending Score (1-100)"],
              color=colors[i],label=f'Cluster{i+1}')
  distances=euclidean_distances(cluster_points,[centroids[i]])
  radius=np.max(distances)
  circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
  plt.gca().add_patch(circle)

plt.scatter(centroids[:,0],centroids[:,1],marker="*",s=200,color='k',label='Centroids')
plt.title("K- means Clustering")
plt.xlabel("Annual Incme (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
```

## Output:
# INSERTED DATASET:
![image](https://github.com/22008650/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/122548204/19b02a04-e57b-4227-9bf3-5af59982a5c5)
# CENTROID AND LABEL VALUES: 
![image](https://github.com/22008650/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/122548204/b8fef697-c39a-4c01-aad1-09cf8cf9ef40)
# CLUSTERING:
![image](https://github.com/22008650/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/122548204/4beabcae-e0a6-463d-a289-8313124c9553)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
