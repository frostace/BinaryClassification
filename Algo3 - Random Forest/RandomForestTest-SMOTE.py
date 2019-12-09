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
from RandomForestFunctions import *

# Initialize useful data
# ===========================================================
df = pd.read_csv('clinvar_conflicting_clean.csv', low_memory=False)
columns_to_change = ['ORIGIN', 'EXON', 'INTRON', 'STRAND', 'LoFtool', 'CADD_PHRED', 'CADD_RAW', 'BLOSUM62']
df[['CLNVI', 'MC', 'SYMBOL', 'Feature_type', 'Feature', 'BIOTYPE', 
 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 
 'BAM_EDIT', 'SIFT', 'PolyPhen']] = df[['CLNVI', 'MC', 'SYMBOL', 'Feature_type', 'Feature', 'BIOTYPE', 
 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 
 'BAM_EDIT', 'SIFT', 'PolyPhen']].fillna(value=0)
df_zero = df.loc[df['CLASS'] == 0]
df_zero = df_zero.sample(n=10000)
df_one = df.loc[df['CLASS'] == 1]
df_one = df_one.sample(n=10000)

df = pd.concat([df_zero, df_one])
df = df.sample(n = df.shape[0])
all_rows = df.values.tolist()
row_num = len(all_rows)

# Divide whole dataset into training set and testing set
# ===========================================================
training_percentage = 0.01  # percent of partition of training dataset
training_size = int(row_num * training_percentage)
testing_size = row_num - training_size
training_attribute = list(df.columns)[: -1]# should exclude 'CLASS'
training_data = all_rows[: training_size]  # training data should include header row
testing_data = all_rows[training_size: ]   # testing data don't need to include header row

# number of attributes to use
# ===========================================================
rand_attribute_subset_len = 5
final_acc_list = []
final_time_list = []
final_attribute = []
forest_size = 1
test_times = 4

for fixed_attribute in training_attribute:
	remaining_attribute = list(training_attribute)[: -1]
	remaining_attribute.remove(fixed_attribute)

	print("Training for: %s" % fixed_attribute)

	start = time.time()

	final_acc = 0

	for pp in range(test_times):

		# Training Random Forest
		# ===========================================================
		random_forest = []
		
		for i in range(forest_size):
			rand_attribute_subset = np.random.choice(a=remaining_attribute, size=rand_attribute_subset_len - 1)
			rand_attribute_subset = np.append(rand_attribute_subset, fixed_attribute)
			# print(rand_attribute_subset)
			training_data = bootstrapped_dataset(all_rows, training_size)
			tree = DecisionTree(training_attribute, rand_attribute_subset, training_data, "CART")
			# tree.print_tree(tree.root)
			random_forest.append(tree)

		end = time.time()
		print("Random Forest Trained! Time: %.03fs" % ((end - start) / 10))

		# Testing Random Forest, Computing TN, TP, FN, FP, etc.
		# ===========================================================

		CMap = {0: 'TN', 1: 'FN', 2: 'FP', 3: 'TP'}
		cutoff = 0.5
		Confusion = {'TN': 0, 'FN': 0, 'FP': 0, 'TP': 0}
		for row in testing_data:
			true_rate_forest = 0
			for tree_i in random_forest:

				# prediction is a counter of label 1 and 0
				pred_counter = tree_i.classify(row, tree_i.root)
				true_rate_tree = pred_counter.get(1, 0) / (pred_counter.get(1, 0) + pred_counter.get(0, 0) + 0.00000001)
				true_rate_forest += true_rate_tree
			true_rate_forest /= forest_size
			true_pred = 1 if true_rate_forest >= cutoff else 0
			indicator = (true_pred << 1) + row[-1]

			# accordingly update confusion matrix
			Confusion[CMap[indicator]] += 1
			
		final_acc += (Confusion['TP'] + Confusion['TN']) / sum(Confusion.values())

	end = time.time()
	print("Random Forest Tested! Time: %.03fs" % ((end - start) / test_times))
	final_acc /= test_times
	print(final_acc, ((end - start) / test_times))
	final_acc_list.append(final_acc)
	final_attribute.append(fixed_attribute)

print(final_acc_list, final_attribute)

