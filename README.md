# meaningful_repository_name
### Chcek unit test coverage
```
$ pip install coverage
$ coverage run --source=./  -m pytest
$ coverage report -m --omit=main.py,tests/*

```


### usage
```
$$ python main.py -i='predict' -v 4 2 1 0
('Inputted values', [4, 2, 1, 0])
[(0.1875, 'setosa'), (0.0625, 'versicolor'), (0, 'virginica')]
PREDICTED TYPE : setosa



$python main.py -i='evaluate' -f='predict.csv'
ACCURACY : 0.88


```