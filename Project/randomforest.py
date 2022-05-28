import Project.main as main
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
import Project.dimred as dimred
from Project.classif_automation import classif_automation
from datetime import datetime

y_train = main.y_train
random_state = main.random_state
results = list()
strategies = [50, 100, 200, 400]
mod_df = main.mod_df
scoring_methods = ['balanced_accuracy', 'precision_weighted', 'recall_weighted',
                   'f1_weighted', 'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
"""
print('start base:',datetime.now())
for scoring_m in scoring_methods:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m', RandomForestClassifier(random_state=random_state, n_jobs=-1))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring=scoring_m, cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(scoring_m, np.mean(n_scores), np.std(n_scores))


print('end base:',datetime.now())
#very small differences in balanced accuracy when ntrees differ (.01 max difference, with average balanced accuracy around .84 )

##CHECK zero_division param to avoid illdefined warning

## PCA_df
print('start pca_df:',datetime.now())
results = list()
df_pca99 = dimred.df_pca99
df_pca90 = dimred.df_pca90
df_pca95 = dimred.df_pca95
df_pca80 = dimred.df_pca80
datasets_pca = [df_pca80, df_pca90, df_pca95, df_pca99]
classif_automation(datasets=datasets_pca, classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                   random_state= random_state, y_train=y_train)
print('end pca_df:',datetime.now())
print('start lda_df:',datetime.now())
## lda_df
results = list()
lda4_df = dimred.lda4_df
lda3_df = dimred.lda3_df
lda2_df = dimred.lda2_df
lda1_df = dimred.lda1_df
datasets_lda= [lda4_df, lda3_df, lda2_df, lda1_df]
classif_automation(datasets=datasets_lda, classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                   random_state= random_state, y_train=y_train)


print('end lda_df:',datetime.now())


## ISO
print('start iso_df', datetime.now())
iso_df = dimred.iso_df
datasets_iso= [iso_df]
classif_automation(datasets=datasets_iso, classifier=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                   random_state= random_state, y_train=y_train)

print('end iso_df:',datetime.now())
"""
"""
# Hyperparameter optimization
print('start grid search', datetime.now())
rfc=RandomForestClassifier(random_state=random_state)
strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,7, 10],
    'criterion' :['gini', 'entropy'],
    'class_weight' :['balanced',"balanced_subsample"]
}
df_pca80 = dimred.df_pca80
scoring_methods=['balanced_accuracy', 'precision_weighted', 'recall_weighted',
                   'f1_weighted', 'roc_auc_ovo_weighted']
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= strat_kfolds, verbose=3, scoring='balanced_accuracy')
CV_rfc.fit(df_pca80, y_train.values.ravel())
print("tuned hyperparameters :(best parameters) ",CV_rfc.best_params_)
print(" balanced accuracy :",CV_rfc.best_score_)
print('end grid search', datetime.now())
"""
"""
NISTF with outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200}
 balanced accuracy : 0.7739616730880612
NISTF without outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 200}
 balanced accuracy : 0.7932156782313887
NISTS with outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 100}
 balanced accuracy : 0.7820371517547614
NISTS without outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 200}
 balanced accuracy : 0.7663963313375078
 
sfinge_default with outliers: 
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200}
 balanced accuracy : 0.9007811715896599
sfinge_default without outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'log2', 'n_estimators': 200}
 balanced accuracy : 0.8843498388272433
 
 sfinge hq with outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200}
 balanced accuracy : 0.9079626239077079
 sfinge hq without outliers:
tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 7, 'max_features': 'sqrt', 'n_estimators': 200}
 balanced accuracy : 0.8888180509377538
 
 sfinge vq with outliers:
 tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 7, 'max_features': 'log2', 'n_estimators': 200}
 balanced accuracy : 0.8415856699423511
 sfingevq without outliers
 tuned hyperparameters :(best parameters)  {'class_weight': 'balanced_subsample', 'criterion': 'entropy', 'max_depth': 4, 'max_features': 'sqrt', 'n_estimators': 200}
 balanced accuracy : 0.8515863066233992
"""
## Final Randomforest
X_test = main.X_test
y_test = main.y_test
df_pca80 = dimred.df_pca80
lda4_df = dimred.lda4_df

df_pca80_test = dimred.df_pca80_test
lda4_df_test = dimred.lda4_df_test

#NIST-F: without outliers RandomForestClassifier(criterion='gini', class_weight='balanced_subsample', max_depth=4,max_features='log2', n_estimators=200)
#NIST_S: with outliers RandomForestClassifier(class_weight='balanced', criterion='entropy', max_depth=4, max_features='sqrt', n_estimators=100)
#Sfing default: with outliers RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=7, max_features='sqrt', n_estimators=200)
#Sfing HQ: with outliers RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=7, max_features='sqrt', n_estimators=200)
#SfinGE_VQ without outliers RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=4, max_features='sqrt', n_estimators=200)

clf = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=4, max_features='sqrt', n_estimators=200)

import pandas as pd
clf.fit(lda4_df, y_train.values.ravel())

pred_clf = clf.predict(lda4_df_test)
y_pred_proba=pd.DataFrame(clf.predict_proba(lda4_df_test))


from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

bal_acc_score = balanced_accuracy_score(y_pred=pred_clf, y_true=y_test)
F = f1_score(y_pred=pred_clf, y_true=y_test, average='weighted')
prec = precision_score(y_pred=pred_clf, y_true=y_test, average='weighted')
recall = recall_score(y_pred=pred_clf, y_true=y_test, average='weighted')
RAUC = roc_auc_score(y_true=y_test[y_test.columns[-1]], y_score=y_pred_proba ,multi_class='ovo', average='weighted')
MCC = matthews_corrcoef(y_pred=pred_clf, y_true=y_test)

print('RF balanced accuracy score is: ', bal_acc_score)
print('precision_weighted is:', prec)
print('recall_weighted is:', recall)
print('F1 score is: ', F)
print('roc_auc_ovo_weighted is:', RAUC)
print('MCC is ', MCC)
"""
NIST-F PCA
RF balanced accuracy score is:  0.8013432838111834
precision_weighted is: 0.8317618704724044
recall_weighted is: 0.826797385620915
F1 score is:  0.8285921680379511
roc_auc_ovo_weighted is: 0.9555442184824765
MCC is  0.7808010901451922
NIST-F LDA
RF balanced accuracy score is:  0.663518418412988
precision_weighted is: 0.7131902227695344
recall_weighted is: 0.696078431372549
F1 score is:  0.7019709562384286
roc_auc_ovo_weighted is: 0.8826042191703369
MCC is  0.6177536089916219

NIST_S PCA
RF balanced accuracy score is:  0.7345364758698091
precision_weighted is: 0.7849066497958078
recall_weighted is: 0.7787878787878788
F1 score is:  0.7797775940138494
roc_auc_ovo_weighted is: 0.9351261010781212
MCC is  0.7197423069506355

NIST-S LDA
RF balanced accuracy score is:  0.6069270482603817
precision_weighted is: 0.6378713327932578
recall_weighted is: 0.6303030303030303
F1 score is:  0.6290493428236609
roc_auc_ovo_weighted is: 0.8628329082066455
MCC is  0.5337996721952637

Sfinge_default PCA:
RF balanced accuracy score is:  0.8998605647766686
precision_weighted is: 0.926719848188445
recall_weighted is: 0.899
F1 score is:  0.9092127259588728
roc_auc_ovo_weighted is: 0.9819344056074473
MCC is  0.8606640400411947

Sfinge Default LDA:
RF balanced accuracy score is:  0.9118290205009607
precision_weighted is: 0.9583443460921545
recall_weighted is: 0.9505
F1 score is:  0.9535540061709568
roc_auc_ovo_weighted is: 0.9891795266772726
MCC is  0.9302108634488828

Sfinge_HQ PCA:
RF balanced accuracy score is:  0.9264844163815005
precision_weighted is: 0.9433182795776172
recall_weighted is: 0.928
F1 score is:  0.9332379327221129
roc_auc_ovo_weighted is: 0.988913397021863
MCC is  0.8994389383941588

Sfinge HQ LDA:
RF balanced accuracy score is:  0.9201527149155938
precision_weighted is: 0.9660125689332321
recall_weighted is: 0.963
F1 score is:  0.9642680317903476
roc_auc_ovo_weighted is: 0.9933892213228823
MCC is  0.9475539742316187

Sfinge_VQ PCA:
RF balanced accuracy score is:  0.9244917452424289
precision_weighted is: 0.9386873961410216
recall_weighted is: 0.9065533980582524
F1 score is:  0.9180409877801038
roc_auc_ovo_weighted is: 0.9849001173170802
MCC is  0.8690847379200808

Sfinge VQ LDA:
RF balanced accuracy score is:  0.9434000665641008
precision_weighted is: 0.9782364106897684
recall_weighted is: 0.9739077669902912
F1 score is:  0.9755617587170295
roc_auc_ovo_weighted is: 0.9934605325831285
MCC is  0.9623756190927595

"""

