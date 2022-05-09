import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

auto_train = pd.read_csv('auto_train.psv', sep='|')
auto_test = pd.read_csv('auto_test.psv', sep='|')

X_train = auto_train['horsepower'].values.reshape(-1, 1)
X_test = auto_test['horsepower'].values.reshape(-1, 1)

y_train = auto_train['mpg']
y_test = auto_test['mpg']

lr = LinearRegression()
lr.fit(X_train, y_train)
coef = lr.coef_[0]
intercept = lr.intercept_

print('training R2:', lr.score(X_train, y_train))
print('test R2:', lr.score(X_test, y_test))

for name, df in [('train', auto_train), ('test', auto_test)]:
    fig = df.plot.scatter(x='horsepower', y='mpg')
    x_min, x_max = fig.get_xlim()

    x_range = np.linspace(x_min, x_max, 1000)
    poly = x_range * coef + intercept
    plt.plot(x_range, poly, color='black')
    plt.show()