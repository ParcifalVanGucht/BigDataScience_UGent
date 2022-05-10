from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import Project.main as main

y_train= main.y_train
#create a new KNN model
knn_cv = KNeighborsClassifier()

results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10]
scaler = StandardScaler()
mod_df = scaler.fit_transform(main.mod_df)
for n in neighbors:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='median')), ('m',KNeighborsClassifier(n_neighbors=n))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(n, np.mean(n_scores), np.std(n_scores))

# n = 3 or 5 gives best balanced accuracy


## Scores for pca dim reduction
results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10, 20, 50, 100]
#
for n in neighbors:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='median')), ('m',KNeighborsClassifier(n_neighbors=n))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, main.df_pca, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(n, np.mean(n_scores), np.std(n_scores))