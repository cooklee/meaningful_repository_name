#!/usr/bin/python

from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel
import sys
import os

# store input arguments
args = sys.argv
n_args = len(args)

if (n_args < 4):
    output = 1
else :
    output = 2
    
    
#print ('Number of arguments:', n_args, 'arguments.')
#print ('Argument List:', str(args))

if (output == 1):
    # new file to load
    file_path = args[2] 
    # load data
    data = CsvReader.create_from_csv(file_path, sep=";")
    
    #new_data = CsvReader.create_from_csv(file_path, sep=";")
    #data = CsvReader.create_from_csv("iris.csv",sep=";")
    
else:
    sepal_length = args[1]
    sepal_width = args[2]
    petal_length = args[3]
    petal_width = args[4]
    # load data
    data = CsvReader.create_from_csv("iris.csv",sep=";")



# data preparation
test_data, to_learn = train_split_data(data, 'iris_type', 0.3)
irisModel = IrisModel(test_data, to_learn, data.get_types_of_data('iris_type'))



#test_data, to_learn = train_split_data(data, 'iris_type', 0.9)
#irisModel = IrisModel(new_data, to_learn, data.get_types_of_data('iris_type'))

irisModel.learn()


if (output == 1):
    print("")
    irisModel.evalution(test_data)

else:
    item_list = [sepal_length, sepal_width, petal_length, petal_width]
    item = [float(i) for i in item_list]
    a = irisModel.predict(item)
    max_prob = max([x[0] for x in a])
    pred_class = [y[1] for y in a if y[0]==max_prob][0]

    print("")
    print ('Input values: sepal length', sepal_length, 'sepal width', sepal_width, 
           "petal length", petal_length, "petal width", petal_width)
    print("Output class:", pred_class)



































































































































































