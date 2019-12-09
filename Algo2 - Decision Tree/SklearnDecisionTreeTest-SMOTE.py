#!/usr/bin/env python

# sklearn lib version
# ===========================================================
from sklearn import tree
from datascience import *
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('fivethirtyeight')
import collections
import math
from tqdm import tqdm
from time import sleep
import pandas as pd
import time
import sys

# print(int(rate))

df = pd.read_csv('clinvar_conflicting_mapped.csv', low_memory=False)
all_rows = df.values.tolist()
columns_backup = df.columns

cate_columns = [0, 2, 3, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 29, 30, 31]


# SMOTE Sampling for unbalanced input data
# ===========================================================
X_train = [row[: -1] for row in all_rows]
y_train = [row[-1] for row in all_rows]
smt = SMOTENC(random_state=42, categorical_features=cate_columns)
X_train, y_train = smt.fit_resample(X_train, y_train)
print("SMOTE-Resampled")

# Overwrite the original all_rows with the re-sampled data
# ===========================================================
all_rows = np.zeros((X_train.shape[0], X_train.shape[1] + 1))
all_rows[:, :X_train.shape[1]] = X_train
all_rows[:, X_train.shape[1]] = y_train
df = pd.DataFrame(all_rows)
df.columns = columns_backup

rate_lower = 0.001
rate_higher = 0.5
rate_lower = float(sys.argv[1])
rate_higher = float(sys.argv[2])
rate_step = 0.001
final_acc_list = []
final_time_list = []
for rate in np.arange(rate_lower, rate_higher + rate_step, rate_step):
	print("Training: rate = %.03f" % rate)
	final_acc = 0

	start = time.time()
	for pp in range(10):
		df = df.sample(n = df.shape[0])
		all_rows = df.values.tolist()
		row_num = len(all_rows)
		training_percentage = rate  # percent of partition of training dataset
		training_size = int(row_num * training_percentage)
		testing_size = row_num - training_size
		training_attribute = list(df.columns)
		training_data = all_rows[: training_size]  # training data should include header row
		testing_data = all_rows[training_size: ]   # testing data don't need to include header row

		X = [row[: -1] for row in training_data]
		Y = [row[-1] for row in training_data]
		clf = tree.DecisionTreeClassifier()
		clf = clf.fit(X, Y)

		test = [row[: -1] for row in testing_data]
		actual_label = [row[-1] for row in testing_data]
		result = clf.predict(test)

		accuracy = 0
		for i in range(len(result)):
		    accuracy += int(result[i] == actual_label[i])
		accuracy /= len(result)
		final_acc += accuracy
	end = time.time()
	final_acc /= 10
	print(final_acc, (end - start) / 10)
	final_acc_list.append(final_acc)
	final_time_list.append((end - start) / 10)
print(final_acc_list, final_time_list)