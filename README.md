# DecisionTreeVisulizer
 To visualize decision tree, random forest, ...






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
    * Generate a random number between 0 and 1, if it falls into the interval of [distribution[i-1], distribution[i]), pick sample i
    * Generate new samples from the original sample set until they are of same size
    * Give all new samples the same weight as before
