import matplotlib.pyplot as plt
import pandas as pd

load_path = 'auto_train.psv'
auto = pd.read_csv(load_path, sep='|')

auto.plot.scatter(x='horsepower', y='mpg')
plt.show()