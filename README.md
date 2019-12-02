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
2. \beta

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
