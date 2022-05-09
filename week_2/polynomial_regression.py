import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

auto_train_train = pd.read_csv('auto_train_train.psv', sep='|')
auto_validation = pd.read_csv('auto_validation.psv', sep='|')

X_train_train = auto_train_train['horsepower'].values.reshape(-1, 1)
y_train_train = auto_train_train['mpg']

X_validation = auto_validation['horsepower'].values.reshape(-1,1)
y_validation = auto_validation['mpg']


for i in range(11):
    print('Current Polynomial degree is:', i)
    pf = PolynomialFeatures()
    X_train_train_poly = pf.fit_transform(X_train_train)
    X_validation_poly = pf.fit_transform(X_validation)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(X_train_train_poly, y_train_train)
    y_pred = lr.predict(X_validation_poly)
    print('Validation error:', mean_squared_error(y_validation, y_pred))