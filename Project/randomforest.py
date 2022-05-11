import Project.main as main
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
import Project.dimred as dimred
from Project.classif_automation import classif_automation

y_train = main.y_train
random_state = main.random_state
results = list()
strategies = [50, 100, 200, 400]
mod_df = main.mod_df
scoring_methods = ['accuracy', 'balanced_accuracy', 'precision_weighted', 'recall_weighted',
                   'f1_weighted', 'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
for scoring_m in scoring_methods:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m', RandomForestClassifier(random_state=random_state, n_jobs=-1))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring=scoring_m, cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(scoring_m, np.mean(n_scores), np.std(n_scores))

#very small differences in balanced accuracy when ntrees differ (.01 max difference, with average balanced accuracy around .84 )

##CHECK zero_division param to avoid illdefined warning

## PCA_df
results = list()
df_pca99 = dimred.df_pca99
df_pca90 = dimred.df_pca90
df_pca95 = dimred.df_pca95
df_pca80 = dimred.df_pca80
datasets = [df_pca80, df_pca90, df_pca95, df_pca99]
classif_automation(datasets=datasets, classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                   random_state= random_state, y_train=y_train)


## lda_df
results = list()
lda_df = dimred.lda_df
datasets = [lda_df]
classif_automation(datasets=datasets, classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                   random_state= random_state, y_train=y_train)
