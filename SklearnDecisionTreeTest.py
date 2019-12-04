#!/usr/bin/env python

# sklearn lib version
# ===========================================================
from sklearn import tree
import pandas as pd
import time
import sys

rate = float(sys.argv[1])
# print(int(rate))

df = pd.read_csv('clinvar_conflicting_mapped.csv', low_memory=False)
# df.head()

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