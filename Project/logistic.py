# evaluate multinomial logistic regression model
import numpy as np
from numpy import mean, std
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import Project.main as main

# define dataset
X_train = main.X_train
y_train = main.y_train

# define the multinomial logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# define the model evaluation procedure
strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# evaluate the model and collect the scores
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=strat_kfolds, n_jobs=-1)
# report the model performance
print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))