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
 
* Given the loss function, how to apply gradient descent? don't know so I used sklearn lib.

* Can I simply consider null entries as 0 inputs?

* Mapping from categorical data to numerical data

* Now that they are all numerical data, i limit their values within [0, 1] for computation cost

* Logistic Performance:

  * Acc = 53.6%

    ![image](https://github.com/frostace/BinaryClassification/blob/master/Algo1%20-%20Logistic%20Regression/Logistic%20ACC.png)

    if we pick 0.4 as cutoff point, Accuracy would be 53.6%.

  * ROC = 0.544

    ![image](https://github.com/frostace/BinaryClassification/blob/master/Algo1%20-%20Logistic%20Regression/Logistic%20ROC.png)

  * AUC slightly larger than 0.5, not so good
