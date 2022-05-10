from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import Project.main as main
import Project.dimred as dimred

random_state = main.random_state
y_train= main.y_train
#create a new KNN model
knn_cv = KNeighborsClassifier()

results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10]
mod_df = main.mod_df
for n in neighbors:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(n, np.mean(n_scores), np.std(n_scores))

# n = 3 or 5 gives best balanced accuracy


## Scores for pca dim reduction
df_pca99 = dimred.df_pca99
df_pca90 = dimred.df_pca90
df_pca95 = dimred.df_pca95
df_pca80 = dimred.df_pca80
results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10, 20, 50, 100]

datasets = [df_pca80, df_pca90, df_pca95, df_pca99]
for dataset in datasets:
    for n in neighbors:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
        # evaluate the model
        strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(pipeline, dataset, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
        # store results
        results.append(n_scores)
        print(len(dataset[0]), n, np.mean(n_scores), np.std(n_scores))
## LDA

results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10, 20, 50, 100]
lda_df = dimred.lda_df
for n in neighbors:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, lda_df, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print('lda',len(lda_df[0]), n, np.mean(n_scores), np.std(n_scores))