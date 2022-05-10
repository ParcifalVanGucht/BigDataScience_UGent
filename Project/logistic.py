import Project.main as main
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
from Project.main import df_pca80, df_pca95, df_pca90, df_pca99, mod_df

# define dataset
X_train = main.X_train
y_train = main.y_train
random_state = main.random_state
## Mod_df
results = list()
scaler = StandardScaler()
mod_df = scaler.fit_transform(mod_df)
scoring_methods = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
for scoring_m in scoring_methods:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='median')), ('m', LogisticRegression(multi_class='multinomial', max_iter=500, random_state=random_state))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring=scoring_m, cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(scoring_m, np.mean(n_scores), np.std(n_scores))

## PCA_df
results = list()
strategies = [50, 100, 200, 400]

datasets = [df_pca80, df_pca90, df_pca95, df_pca99]
scoring_methods = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
for dataset in datasets:
    print(dataset)
    for scoring_m in scoring_methods:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('i', SimpleImputer(strategy='median')), ('m', LogisticRegression(multi_class='multinomial', max_iter=500, random_state=random_state))])
        # evaluate the model
        strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(pipeline, dataset, y_train.values.ravel(), scoring=scoring_m, cv=strat_kfolds, n_jobs=-1)
        # store results
        results.append(n_scores)
        print(scoring_m, np.mean(n_scores), np.std(n_scores))