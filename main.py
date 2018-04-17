from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel

import sys
#import os
#dir_path = os.path.dirname(os.path.realpath(__file__))

#print("!!!!!!"+dir_path)

# load data
data = CsvReader.create_from_csv("iris.csv", sep=";")

# data preparation
test_data, to_learn = train_split_data(data, 'iris_type', 0.3)

irisModel = IrisModel(test_data, to_learn, data.get_types_of_data('iris_type'))

irisModel.learn()

for item in test_data:
    a = irisModel.predict(item)
    b =1
#



args = sys.argv[1:]

if args[0] == '--help':
    print("Instruction:")
    print("-A filename: to print test accuracy for [filename]")
    print("-P float float float float: to print type for [float float float float]")
elif args[0] == '-T':
    print("Evaluation from training data:")
    irisModel.evalution(test_data)
elif args[0] == '-P':
    if len(args) != 5:
        print("Wrong arguments")
        exit(-1)
    preds = irisModel.predict([float(arg.replace(',', '.')) for arg in args[1:]])
    pred_type = 'Not even a flower...'
    score = 0
    for s, t in preds:
        if s > score:
            pred_type = t
            score = s
    print("The classification is: " + pred_type)
elif args[0] == '-A':
    if len(args) != 2:
        print("Wrong arguments")
        exit(-1)
    input_data = CsvReader.create_from_csv(args[1], sep=";")
    print("Evaluation from input data:")
    irisModel.evalution(test_data)








































































































































































