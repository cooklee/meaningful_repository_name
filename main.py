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


if __name__ == "__main__":

    flag = sys.argv[1]

    if flag == "-f":
        input_file = sys.argv[2]
        print('inp',input_file)
        input_data = CsvReader.create_from_csv(input_file, sep=";")
        for item in input_data:
            print(final_model.predict(item))
    elif flag=="-a":
        input_data = [float(x.replace(',','.')) for x in sys.argv[2:]]
        print(final_model.predict(input_data))


