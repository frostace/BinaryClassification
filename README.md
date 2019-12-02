# DecisionTreeVisulizer
 To visualize decision tree, random forest, ...

# Objectives:
1.	Estimate how many variants will have conflicting classifications, why are they considered to have conflicting classifications?
	Methodology: Logistic Regression
2.	Evaluate the importance of all these features, find the most important one and analyze why is it so important.
	Methodology: Decision Tree, Bootstrap (and probably MLE)
3.	Redo Question 1 with classification label unknown, compare the 2 results.
	Methodology: K-means Clustering

## Logistic Regression:
### Notation:
1. X: n-dimentional input characteristics
2. \beta:
3. \hat y_i: 
4. y_i: 

### Intro:
1. Given Linear Regression Model: 
	* yita_i = beta_0 + beta_1 * x_1i + ... + beta_p * x_pi
2. Link Function: 
	* g(mu_i) = yita_i
	* where, mu_i is our outcome
	* Describes how the mean E(Y_i) = mu_i depends on the linear predictor
3. Binary Classification Case:
	* link function must map from (0, 1) to (-inf, inf)
	* logit function: g(mu_i) = logit(mu_i) = log(mu_i / (1 - mu_i))
	* probit function: g(mu_i) = fai^(-1)(mu_i)
		* where, fai() is the CDF of the standard normal distribution.

### Logistic Model:
1. Model the probability that Y equals 1, given X:
	* p(X) = P(Y = 1 | X)
	* logit(p(X)) = log(p(X) / (1 - p(X))) = beta.T * X
	* p(X) = exp(beta.T * X) / (1 + exp(beta.T * X))
	* p(X), namely \hat y_i, is the probability

### Loss Function:
1. Loss Function:
	* Loss = sigma_i[(-y_i * log(\hat y_i)) * -(1 - y_i)log(1 - \hat y_i)]

2. Maximum Likelihood Estimator (MLE):
	* Minimizing the last loss function equals to maximizing (\hat y_i)^y_i * (1 - \hat y_i)^(1 - y_i)
	* which exactly refers to the probability of observing y_i when y_i follows the Bernoulli distribution

### Performance Evaluation:
1. Decision Boundary:
	* our predicted outcome \hat y_i is a number between 0 and 1, we need another mapping to map the region to a binary output, thus, a decision boundary (cutoff point).
	* we usually use 0.5 as a cutoff point blindly.

2. Evaluation:
	* Confusion Matrix:
		First go true or not, then go actual value (observed value)
		* TP: Prediction is True + Actual value is Positive
		* FP: Prediction is False + Actual value is Positive
		* TN: Prediction is True + Actual value is Negative
		* FN: Prediction is False + Actual value is Negative
	* Accuracy:
		* Accuracy = (TP + TN) / (TP + FN + FP + TN)
		* Sensitivity = TP / (TP + FN)
		* Specificity = TN / (TN + FP)

3. ROC Curve:
	* for all possible cutoff points, compute FP and TP, plot every coords: (FP, TP) on a figure
	* Null Model:
		randomly assign a prediction above the cutoff point as True / False
		randomly assign a prediction below the cutoff point as True / False
		So, FP = TP
	* AUC (Area under the ROC Curve)
		* the steeper the ROC Curve, the greater the predictive power
		* AUC = 0.5 -> no predictive power
		* AUC = 1 -> perfect predictive power

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
