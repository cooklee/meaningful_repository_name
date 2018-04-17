from csv_reader import CsvReader
from DataScientistMethod import train_split_data
from model import IrisModel
import argparse
import os
from operator import truediv
parser = argparse.ArgumentParser(description='predicts flower type from input(s)')
parser.add_argument('-i', '--input_type', type=str, required=True, choices=['predict', 'evaluate'],
                    help='functionality to run')
parser.add_argument('-f', '--file_path', type=str, required=False,
                    help='absolute file path for evaluation')
parser.add_argument('-v', '--measurements', type=int, nargs='+', required=False,
                    help='integer input measurements for prediction : '
                         'sepal length (cm) ,sepal width (cm) ,petal length (cm),petal width (cm)')

args = parser.parse_args()


def predict_from_measurements(measurements):
    sepal_length, sepal_width, petal_length, petal_width = measurements[0], measurements[1], measurements[2], \
                                                           measurements[3]

    print('Inputted values',measurements)

    # load data
    train_data = CsvReader.create_from_csv("iris.csv", sep=";")
    data, to_learn= train_split_data(train_data, 'iris_type', 0)
    irisModel = IrisModel(data,to_learn, train_data.get_types_of_data('iris_type'))
    irisModel.learn()
    measurements.append('na')
    prediction = irisModel.predict(measurements)
    predicted_type = max(prediction, key=lambda item: item[0])[1]
    print(prediction)
    print('PREDICTED TYPE : {}'.format(predicted_type))


def evaluate_from_file(file_path):
    # load data
    train_data = CsvReader.create_from_csv('iris.csv', sep=";")
    data,to_learn= train_split_data(train_data, 'iris_type', 0)
    irisModel = IrisModel(data,to_learn, train_data.get_types_of_data('iris_type'))
    irisModel.learn()


    test_data = CsvReader.create_from_csv(file_path, sep=";")
    accuracies = []
    for test_item in test_data:
        prediction = irisModel.predict(test_item)
        predicted_type = max(prediction, key=lambda item: item[0])[1]
        if predicted_type == test_item[-1]:
            accuracies.append(1)
        else:
            accuracies.append(0)
    print('ACCURACY : {:.2f}'.format(truediv(sum(accuracies), len(accuracies))))


print(args.input_type)
print(args.measurements)
print(args.file_path)

if args.input_type == 'predict':
    predict_from_measurements(args.measurements)
elif args.input_type == 'evaluate':
    evaluate_from_file(args.file_path)
