#!/usr/bin/env python

# ===========================================================
from sklearn import tree
import pandas as pd
import time
import sys
# Import lib
# ===========================================================
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

rate = float(sys.argv[1])
# print(int(rate))

# Initialize useful data
# ===========================================================
# with open('clinvar_conflicting_clean.csv', 'r') as f:
#     reader = csv.reader(f)
#     temp_rows = list(reader)
df = pd.read_csv('clinvar_conflicting_clean.csv', low_memory=False)
columns_to_change = ['ORIGIN', 'EXON', 'INTRON', 'STRAND', 'LoFtool', 'CADD_PHRED', 'CADD_RAW', 'BLOSUM62']
df[['CLNVI', 'MC', 'SYMBOL', 'Feature_type', 'Feature', 'BIOTYPE', 
 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 
 'BAM_EDIT', 'SIFT', 'PolyPhen']] = df[['CLNVI', 'MC', 'SYMBOL', 'Feature_type', 'Feature', 'BIOTYPE', 
 'cDNA_position', 'CDS_position', 'Protein_position', 'Amino_acids', 'Codons', 
 'BAM_EDIT', 'SIFT', 'PolyPhen']].fillna(value="null")

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
	for cutoff in np.arange(0, 1 + step_size, step_size):
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
	    # concatenate the confusion matrix values into the overall ROC Table
	    thisline = [cutoff] + list(Confusion.values()) + [(Confusion['TP'] + Confusion['TN']) / sum(Confusion.values())]
	    ROC = ROC.with_row(thisline)
	ROC = ROC.with_column('SENSITIVITY', ROC.apply(lambda TP, FN: TP / (TP + FN + 0.00000001), 'TP', 'FN'))
	ROC = ROC.with_column('FPR', ROC.apply(lambda TN, FP: FP / (TN + FP + 0.00000001), 'TN', 'FP'))
	ROC = ROC.with_column('FMEAS', ROC.apply(lambda TP, FP, FN: 2 * (TP / (TP + FN + 0.00000001)) * (TP / (TP + FP + 0.00000001)) / (TP / (TP + FN + 0.00000001) + TP / (TP + FP + 0.00000001) + 0.00000001), 'TP', 'FP', 'FN'))

	# Original Testing
	# ===========================================================

	accuracy = []
	for row in testing_data:
	    classification = tree.classify(row, tree.root)
	    if len(classification) == 1:
	        accuracy.append(int(classification.get(row[-1], 0) > 0))
	    else:
	        tot = sum(classification.values())
	        accuracy.append(classification.get(row[-1], 0) / tot)
	final_acc += sum(accuracy) / len(accuracy)

end = time.time()
print("Decision Tree Trained! Time: %.03fs" % ((end - start) / 10))
final_acc /= 10
print(final_acc, ((end - start) / 10))