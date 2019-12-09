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

2. Adjust size of the random subset of features used
	* Estimate the accuracy of the random forest with out-of-bag data 
	* Change the number of features used per step 
	* Typically, we start by using the sqrt of the number of variables and try a few settings above and below that value 

[Ref] (https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)


### Performance
* Acc = 68.6%
	
![image](https://github.com/frostace/BinaryClassification/blob/master/Algo3%20-%20Random%20Forest/Random%20Forest%20ACC.png)

* ROC -> AUC = 0.747

![image](https://github.com/frostace/BinaryClassification/blob/master/Algo3%20-%20Random%20Forest/Random%20Forest%20ROC.png)


### bug track
1. sometimes, i run many times for the random forest, the acc remains the same, which is weird, it should at least change a little bit.
	* overfitting leads to cheating, adjust the convergence level
