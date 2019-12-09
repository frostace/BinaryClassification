# K means

Steps: 

1. Select the number of clusters you want to identify in your data 
2. Randomly select k distinct data points 
3. Measure the distance between the 1st point and the k initial clusters 
4. Assign the 1st point to the nearest cluster 
5. Iterate through all points and do step 3 & 4 
6. Calculate the mean of each cluster 
7. Use the calculated mean of each cluster as k new initial data points and restart from 3 
8. Loop until the mean converge 
9. Do Step 1 - 8 for n times, select the best one 

How to select K? 

- Try various K values, evaluate the performance by computing the total variation of the clusters 
- Plot the Reduction of variation - #number of clusters figure, and find the elbow point and its corresponding K 

Ref: https://www.youtube.com/watch?v=4b5d3muPQmA


### Performance
* Acc = 60% | cutoff = 0.4

![image](https://github.com/frostace/BinaryClassification/blob/master/Algo5%20-%20K%20means/K%20means%20ACC.png)

* ROC -> MESSY

![image](https://github.com/frostace/BinaryClassification/blob/master/Algo5%20-%20K%20means/K%20means%20ROC.png)

  * why not working
  * b.c. i mapped the categorical data to numerical data, then i disturbed the categorical similarity and replaced it with a random numerical similarity related with their indices.
