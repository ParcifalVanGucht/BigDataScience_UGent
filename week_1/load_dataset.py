import pandas as pd

load_path = 'week_1/iris.psv'

iris = pd.read_csv(load_path, sep='|')

print(iris.head(10))
print('Success!')
