#!/usr/bin/env python

# Import lib
# ===========================================================
from sklearn import tree
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTENC
import pandas as pd
import time
import sys
import csv
from datascience import *
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
from DecisionTreeFunctions import *

# rate = float(sys.argv[1])
# print(int(rate))

# Initialize useful data
# ===========================================================
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
		# Training
		# ===========================================================
		
		df = df.sample(n = df.shape[0])
		all_rows = df.values.tolist()
		row_num = len(all_rows)

		training_percentage = rate  # percent of partition of training dataset
		training_size = int(row_num * training_percentage)
		testing_size = row_num - training_size
		training_attribute = list(df.columns)
		training_data = all_rows[: training_size]  # training data should include header row
		testing_data = all_rows[training_size: ]   # testing data don't need to include header row

		tree = DecisionTree(training_attribute, training_data, "CART")
		

		# Testing and Computing TN, TP, FN, FP, etc. 
		# ===========================================================
		ROC = Table(make_array('CUTOFF', 'TN', 'FN', 'FP', 'TP', 'ACC'))
		step_size = 0.05
		CMap = {0: 'TN', 1: 'FN', 2: 'FP', 3: 'TP'}
		# 00(0) -> TN
		# 01(1) -> FN
		# 10(2) -> FP
		# 11(3) -> TP
		cutoff = 0.5
		# for cutoff in np.arange(0, 1 + step_size, step_size):
		Confusion = {'TN': 0, 'FN': 0, 'FP': 0, 'TP': 0}
		for row in testing_data:
			# prediction is a counter of label 1 and 0
			pred_counter = tree.classify(row, tree.root)
			true_rate = pred_counter.get(1, 0) / (pred_counter.get(1, 0) + pred_counter.get(0, 0) + 0.00000001)
			#         print(true_rate)
			true_pred = 1 if true_rate >= cutoff else 0
			indicator = (true_pred << 1) + row[-1]
			# accordingly update confusion matrix
			Confusion[CMap[indicator]] += 1
		#     # concatenate the confusion matrix values into the overall ROC Table
		#     thisline = [cutoff] + list(Confusion.values()) + [(Confusion['TP'] + Confusion['TN']) / sum(Confusion.values())]
		#     ROC = ROC.with_row(thisline)
		# ROC = ROC.with_column('SENSITIVITY', ROC.apply(lambda TP, FN: TP / (TP + FN + 0.00000001), 'TP', 'FN'))
		# ROC = ROC.with_column('FPR', ROC.apply(lambda TN, FP: FP / (TN + FP + 0.00000001), 'TN', 'FP'))
		# ROC = ROC.with_column('FMEAS', ROC.apply(lambda TP, FP, FN: 2 * (TP / (TP + FN + 0.00000001)) * (TP / (TP + FP + 0.00000001)) / (TP / (TP + FN + 0.00000001) + TP / (TP + FP + 0.00000001) + 0.00000001), 'TP', 'FP', 'FN'))

		final_acc += (Confusion['TP'] + Confusion['TN']) / sum(Confusion.values())

	end = time.time()
	print("Decision Tree Trained! Time: %.03fs" % ((end - start) / 10))
	final_acc /= 10
	print(final_acc, ((end - start) / 10))
	final_acc_list.append(final_acc)
	final_time_list.append((end - start) / 10)
print(final_acc_list, final_time_list)