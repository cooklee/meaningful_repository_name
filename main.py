from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel

# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))

# print("!!!!!!"+dir_path)

# load data
data = CsvReader.create_from_csv("iris.csv", sep=";")

# data preparation
test_data, to_learn = train_split_data(data, 'iris_type', 0.3)

irisModel = IrisModel(test_data, to_learn, data.get_types_of_data('iris_type'))
irisModel.learn()

for item in test_data:
    # print(item)
    a = irisModel.predict(item)
    # print(a)

print(irisModel.evalution(test_data))
