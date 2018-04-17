#!/usr/bin/python

from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel
import sys
import os

# store input arguments
args = sys.argv

print 'Number of arguments:', len(args), 'arguments.'
print 'Argument List:', str(args)

file_path = args[0]
sepal_length = args[1]
sepal_width = args[2]
petal_length = args[3]
petal_width = args[4]


dir_path = os.path.dirname(os.path.realpath(file_path))
print("!!!!!!"+dir_path)

# load data
data = CsvReader.create_from_csv(file_path, sep=";")
#data = CsvReader.create_from_csv("iris.csv",sep=";")


# data preparation
test_data, to_learn = train_split_data(data, 'iris_type', 0.3)

irisModel = IrisModel(test_data, to_learn, data.get_types_of_data('iris_type'))

irisModel.learn()


item = [sepal_length, sepal_width, petal_length, petal_width]
a = irisModel.predict(item)
max_prob = max([x[0] for x in a])
pred_class = [y[1] for y in a if y[0]==max_prob]

print("predicted class:", pred_class)

    
"""
for item in test_data:
    a = irisModel.predict(item)
    max_prob = max([x[0] for x in a])
    pred_class = [y[1] for y in a if y[0]==max_prob]
"""

#irisModel.evalution(test_data)








































































































































































