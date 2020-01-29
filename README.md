Visualization Ref: https://explained.ai/decision-tree-viz/

# Project Objectives:
1. Feature Engineering
	* Evaluate the importance of all these features, find the most important one and analyze why is it so important.
   	* Methodology: Random Forest
	
2. Classification
	* Estimate how many variants will have conflicting classifications, why are they considered to have conflicting classifications?
	* Methodology: Logistic Regression, Decision Tree, Random Forest, Adaboost

3. Clustering
	* With label unknown, compare the 2 results.
  	* Methodology: K-means Clustering

# Dataset
https://www.kaggle.com/kevinarvai/clinvar-conflicting

# Performance Evaluation:
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
	* Performance:
		* Accuracy = $\frac{TP + TN}{TP + FN + FP + TN}$
		* Sensitivity (TPR) = $\frac{TP}{TP + FN}$
		  * Sensitivity/recall – how good a test is at detecting the positives. A test can cheat and maximize this by always returning “positive”.
		* Specificity (1 - FPR) = $\frac{TN}{TN + FP}$
		  * Specificity – how good a test is at avoiding false alarms. A test can cheat and maximize this by always returning “negative”.
		* Precision = $\frac{TP}{TP + FP}$
		  * Precision – how many of the positively classified were true. A test can cheat and maximize this by only returning positive on one result it’s most confident in.
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

# Outcome
![image](https://github.com/frostace/BinaryClassification/blob/master/Pres/Accuracy%20-%20Cutoff%20Comparison.png)

![image](https://github.com/frostace/BinaryClassification/blob/master/Pres/ROC%20Curve%20Comparison.png)

# Documentation

Read more about

  * [Logistic Regression](Algo1 - Logistic Regression/README.md)
  * [Decision Tree](Algo2 - Decision Tree/README.md)
  * [Random Forest](Algo3 - Random Forest/README.md)
  * [Adaboost](Algo4 - Adaboost/README.md)
  * [K Means](Algo5 - K means/README.md)
 
# Contributing

You can [Contribute](docs/contributing.md) to this project with issues or pull requests.

# Release Notes

See [RELEASE NOTES](RELEASE_NOTES.md) file.

# License

See [MIT LICENSE](https://github.com/frostace/BinaryClassification/blob/master/LICENSE) file.

# Contact

If you have any ideas, feedback, requests or bug reports, you can reach me at
[frostace0723@gmail.com](mailto:frostace0723@gmail.com)
