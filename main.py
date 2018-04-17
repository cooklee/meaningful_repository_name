from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

accuracy = 0
threshold = 0.98

while  (accuracy < threshold ):

    data = CsvReader.create_from_csv("iris.csv", sep=";")

    test_data, to_learn = train_split_data(data, 'iris_type', 0.3)

    irisModel = IrisModel(test_data, to_learn, data.get_types_of_data('iris_type'))
    irisModel.learn()

    accuracy, final_model = irisModel.evalution(test_data)

print('Best Model Found', accuracy )

