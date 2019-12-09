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
