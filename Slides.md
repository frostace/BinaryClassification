# DecisionTreeVisulizer
 To visualize decision tree, random forest, ...

# Project Objectives:
1.	Estimate how many variants will have conflicting classifications, why are they considered to have conflicting classifications?
	Methodology: Logistic Regression
	
	* Given the loss function, how to apply gradient descent? don't know so I used sklearn lib.
	
	* Can I simply consider null entries as 0 inputs?
	
	* Mapping from categorical data to numerical data
	
	* Now that they are all numerical data, i limit their values within [0, 1] for computation cost
	
	* Logistic Performance:
	
	  * Acc
	
	    ![Logistic ACC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Logistic ACC.png)
	
	    if we pick 0.95 as cutoff point, Accuracy would be 74.7%.
	
	  * ROC
	
	    ![Logistic ROC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Logistic ROC.png)
	
	  * AUC = 0.538, slightly larger than 0.5, not so good
	
	* Use this result as a benchmark, let's move on and see if we can do better.
	
	  * ACC = 74.7%
	  * AUC = 0.538
	
2.	Evaluate the importance of all these features, find the most important one and analyze why is it so important.
  Methodology: Decision Tree, Bootstrap (and probably MLE)
  
  * We can tell apart a dog from a cat immediately with a glance, but what if we cannot see the whole picture, we are just allowed to ask yes or no questions??
  
  * This is basically the main idea of decision tree.
  
  * Discuss a little about find_best_question, give program flow chart
  
  * Discuss Simple Decision tree performance
  
    * Acc
  
      ![Decision Tree ACC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Decision Tree ACC.png)
  
    * ROC -> AUC
  
      Problem: Looks good, but cutoff is not working, try printing true rate values
      
      Possible reasons:
      
	    1. ~~Overfitting~~, i changed the gini convergence condition to gini <= 0.01
      
      ![Decision Tree ROC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Decision Tree ROC.png)
      
      2. Unbalanced input data
      
         ​	F-measure = 0.284839
      
      ![label_distribution](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/label_distribution.png)
      
	  * Performance Comparison
      
      ![Decision Tree Time & Acc by Training percentage](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Decision Tree Time & Acc by Training percentage.png)![Decision Tree Time & Sklearn Acc by Training percentage](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Decision Tree Time & Sklearn Acc by Training percentage.png)
    
      
    
    * Overall Performance
      
      * ACC = 75%
    
      * AUC = 0.642
    
  * Improve with bootstrapping: Random Forest
  
    * Acc
  
      ![Random Forest ACC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Random Forest ACC.png)
  
    * ROC -> AUC
  
      ![Random Forest ROC](/Users/frostace/Documents/GitHub/DecisionTreeVisulizer/Random Forest ROC.png)
  
  * Improve with Adaboost:
  
    * Compare Performance
  
3. Redo Question 1 with classification label unknown, compare the 2 results.
  Methodology: K-means Clustering
  
  ```markdown
  training_size: 65188
  winner:  48053 17135
  Acc:  0.6217555378290482
  Counter({0.0: 36075, 1.0: 11978}) Counter({0.0: 12679, 1.0: 4456})
  ```
  
  Acc = 62.2%

## Logistic Regression:

### Notation:
1. X: n-dimentional input characteristics
2. $\beta$: Coefficients in our linear regression model
3. $y_i$: Actuall (Observed)  Outcome of an input $X_i$
4. $\hat y_i$: probability of observing the outcome $y_i$

### Intro:
1. Given Linear Regression Model: 
	* $\eta_i = \beta_0 + \beta_1 x_{1i} + ... + \beta_p x_{pi}$
2. Link Function: 
	* $g(\mu_i) = \eta_i$
	* where, mu_i is our outcome
	* Describes how the mean $E(Y_i) = \mu_i$ depends on the linear predictor
3. Binary Classification Case:
	* link function must map from (0, 1) to (-inf, inf)
	* logit function: $g(\mu_i) = logit(\mu_i) = log_e(\frac{\mu_i}{1 - \mu_i})$
	* probit function: $g(\mu_i) = \phi^{-1}(\mu_i)$
		* where, fai() is the CDF of the standard normal distribution.

### Logistic Model:
1. Model the probability that Y equals 1, given X:
	* p(X) = P(Y = 1 | X)
	* $logit(p(X)) = log_e(\frac{p(X)}{1 - p(X)}) = \beta^T X$
	* p(X) = $\frac{e^{\beta^T X}}{1 + e^{\beta^T X}}$
	* p(X), namely $\hat y_i$, is the probability of observing $y_i$

### Loss Function:
1. Loss Function:
	* $J(\beta) = \sum_i[(-y_i * log(\hat y_i)) * -(1 - y_i)log(1 - \hat y_i)]$

2. Maximum Likelihood Estimator (MLE):
	* Minimizing the last loss function equals to maximizing $(\hat y_i)^{y_i} * (1 - \hat y_i)^{1 - y_i}$
	* which exactly refers to the probability of observing y_i when y_i follows the Bernoulli distribution

### Performance Evaluation:
1. Decision Boundary:
	* our predicted outcome \hat y_i is a number between 0 and 1, we need another mapping to map the region to a binary output, thus, a decision boundary (cutoff point).
	* we usually use 0.5 as a cutoff point blindly.

2. Evaluation:
	* Confusion Matrix:
		First go true or not, then go actual value (observed value)
		* TP: Prediction is True + Predicted value is Positive
		* FP: Prediction is False + Predicted value is Positive
		* TN: Prediction is True + Predicted value is Negative
		* FN: Prediction is False + Predicted value is Negative
	* Accuracy:
		* Accuracy = $\frac{TP + TN}{TP + FN + FP + TN}$
		* Sensitivity (TPR) = $\frac{TP}{TP + FN}$
		* Specificity (1 - FPR) = $\frac{TN}{TN + FP}$
	* Precision = $\frac{TP}{TP + FP}$
		* Recall = $\frac{TP}{TP + FN}$
		* F-measure = $\frac{2Precesion\times Recall}{Precesion + Recall}$
	
3. ROC Curve:
	* for all possible cutoff points, compute Specificity and Sensitivity, plot every coords: (1 - Specificity, Sensitivity) on a figure
	* Null Model:
		randomly assign a prediction above the cutoff point as True / False
		randomly assign a prediction below the cutoff point as True / False
		So, FP = TP
	* AUC (Area under the ROC Curve)
		* the steeper the ROC Curve, the greater the predictive power
		* AUC = 0.5 -> no predictive power
		* AUC = 1 -> perfect predictive power

## Random Forest

1. Building a Random Forest: 

  * Step1: 
  	* Create a bootstrapped dataset 
  * Step2: 
  	* Create a decision tree with the bootstrapped dataset, but only use a random subset of variables (or columns) at each step 
  	* We are randomly pick x variables, or randomly pick variables from a given set of x variables???
  	  * I think it's the latter
  * Step3: 
    * Return to Step1 

1. Estimate the accuracy of the random forest with out-of-bag data 
2. Change the number of variables used per step 
3. - Typically, we start by using the sqrt of the number of variables and try a few settings above and below that value 

[Ref] (https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

## Adaboost

3 ideas 

1. Combination of weak learners 

	* Adaboost combines a lot of weak learners to make classifications 
	* Weak learners are almost always stumps 

1. Weight 

	* Some stumps get more say in the classification than others 

1. Dependency on Previous mistakes 

	* Each stump is made by taking previous stump’s mistakes into account 

Steps: 

1. Initialize a panda dataframe with equal sample weights 

2. Iterate through all attributes to generate a decision stump for each attribute 

3. See which one makes the smallest total error, pick this as our first decision stump 

4. Update sample weight as following principle: 

  * Amount_of_say = $0.5 * ln(\frac{1 - total error}{total error}) $

  * For those which we have misclassified, new_weight = ori_weight * e^(Amount_of_say) 

  * For those which we have correctly classified, new_weight = ori_weight * e^(-Amount_of_say) 

  * Normalize the new sample weight column 

5. Resample 

  * Use the prefix sum of the new sample weight as a distribution. (e.g. [0.07, 0.07, 0.49, ...] -> [0.07, 0.14, 63, ...] 

  * Generate a random number between 0 and 1, if it falls into [distribution[i-1], distribution[i]), pick sample i 

  * Generate new samples from the original sample set until they are of same size 

  * Give all new samples the same weight as before 

Ref: https://www.youtube.com/watch?v=LsK-xG1cLYA 

## K means

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

### Overall Bugs:

1. there exist a situation s.t. a question can increase the gini info instead of decreasing it or at least keeping it remain
2. ~~find_best_question function is considering 'CLASS' column as an attribute to try to raise a question~~
3. ~~when raising a question, we should skip meaningless reference values like 'null' and 'nan', etc.~~
4. when answering a question, say, tring to compare 'null' to a certain reference value, we should roll a dice to decide, which branch to go.


## Simple Decision Tree
[[0.001, 0.7233235569614422, 1.5288949012756348], 
[0.002, 0.6979157059854284, 5.077696084976196], 
[0.003, 0.7215392426876741, 6.001860857009888], 
[0.004, 0.7003295958600295, 9.227790832519531], 
[0.005, 0.7059494627137197, 12.09618592262268], 
[0.006, 0.7055882216769295, 17.873859167099], 
[0.007, 0.7105604646851634, 34.53915524482727], 
[0.008, 0.7150169329024078, 55.14473509788513], 
[0.009, 0.6929816414352497, 52.855583906173706], 
[0.01, 0.7081829028309342, 66.66273093223572]]

## Random Forest

### bugs:
1. sometimes, i run many times for the random forest, the acc remains the same, which is weird, it should at least change a little bit.

## Adaboost
3 ideas
1. Combination of weak learners
    * Adaboost combines a lot of weak learners to make classifications
    * Weak learners are almost always stumps
2. Weight
    * Some stumps get more say in the classification than others
3. Dependency on Previous mistakes
    * Each stump is made by taking previous stump’s mistakes into account

Steps:
1. Initialize a panda dataframe with equal sample weights
2. Iterate through all attributes to generate a decision stump for each attribute
3. See which one makes the smallest total error, pick this as our first decision stump
4. Update sample weight as following principle:
    * Amount_of_say = 0.5 * ln((1 - total_error) / total_error)
    * For those which we have misclassified, new_weight = ori_weight * e^(Amount_of_say)
    * For those which we have correctly classified, new_weight = ori_weight * e^(-Amount_of_say)
    * Normalize the new sample weight column
5. Resample
    * Use the prefix sum of the new sample weight as a distribution. (e.g. [0.07, 0.07, 0.49, ...] -> [0.07, 0.14, 63, ...]
    * Generate a random number between 0 and 1, if it falls into [distribution[i-1], distribution[i]), pick sample i
    * Generate new samples from the original sample set until they are of same size
    * Give all new samples the same weight as before
