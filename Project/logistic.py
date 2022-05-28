import Project.main as main
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
import Project.dimred as dimred
from Project.classif_automation import classif_automation
from datetime import datetime

# define dataset
X_train = main.X_train
y_train = main.y_train
random_state = main.random_state
## Mod_df
results = list()
mod_df = main.mod_df
scoring_methods = ['balanced_accuracy', 'precision_weighted', 'recall_weighted',
                   'f1_weighted', 'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
print('start base model', datetime.now())

for scoring_m in scoring_methods:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m', LogisticRegression(multi_class='multinomial',
                                                        max_iter=1200, random_state=random_state))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(),
                               scoring=scoring_m, cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(scoring_m, np.mean(n_scores), np.std(n_scores))
print('end base model', datetime.now())
## PCA_df
df_pca99 = dimred.df_pca99
df_pca90 = dimred.df_pca90
df_pca95 = dimred.df_pca95
df_pca80 = dimred.df_pca80
datasets = [df_pca80, df_pca90, df_pca95, df_pca99]
print('start pca model', datetime.now())
classif_automation(datasets=datasets, classifier=LogisticRegression(multi_class='multinomial', max_iter=1200,
                                                        random_state=random_state),
                   y_train=y_train,
                   random_state=random_state)
print('end pca model', datetime.now())
## lda_df
lda4_df = dimred.lda4_df
lda3_df = dimred.lda3_df
lda2_df = dimred.lda2_df
lda1_df = dimred.lda1_df
datasets= [lda4_df, lda3_df, lda2_df, lda1_df]
classif_automation(datasets=datasets, classifier=LogisticRegression(multi_class='multinomial', max_iter=1200,
                                                        random_state=random_state),
                   y_train=y_train,
                   random_state=random_state)
print('end lda classif df:', datetime.now())

##iso_df

iso_df = dimred.iso_df

datasets= [iso_df]
classif_automation(datasets=datasets, classifier=LogisticRegression(multi_class='multinomial', max_iter=1200,
                                                        random_state=random_state),
                   y_train=y_train,
                   random_state=random_state)
print('end iso classif df:', datetime.now())


#HYPERPARAM
# Grid search cross validation
print('start grid search for hyperparameter optimization', datetime.now(), '\n')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l2", 'none']}# l1 lasso l2 ridge
logreg=LogisticRegression(multi_class='multinomial', max_iter=1200,
                                                        random_state=random_state)
strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
logreg_cv=GridSearchCV(logreg,grid,cv=strat_kfolds, scoring='balanced_accuracy')
logreg_cv.fit(df_pca80,y_train.values.ravel())

print("tuned hyperparameters :(best parameters) ",logreg_cv.best_params_)
print(" balanced accuracy :",logreg_cv.best_score_)
print('\n end', datetime.now())

"""
Grid search for pca80:
NIST-F with outliers:
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.8434302334365793
 NIST-F without outliers 
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.8362080631025295

NIST-S with outliers
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.8216344253901102
NIST-S without outliers
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.8157972790824761

SFING_default with outliers:
tuned hyperparameters :(best parameters)  {'C': 0.1, 'penalty': 'l2'}
 balanced accuracy : 0.8975788461258567
SFING_default without outliers:
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.8898495461710008

Sfing_hq with outliers:
tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.9127497037911058
Sfing_hq without 1384 outliers:
tuned hyperparameters :(best parameters)  {'C': 0.1, 'penalty': 'l2'}
 balanced accuracy : 0.9080178078173733
 
 SFING_VQ with outliers:
 tuned hyperparameters :(best parameters)  {'C': 1.0, 'penalty': 'l2'}
 balanced accuracy : 0.8355747263806641
 SFING_VQ without 2706 outliers:
 tuned hyperparameters :(best parameters)  {'C': 0.01, 'penalty': 'l2'}
 balanced accuracy : 0.861576640033951
"""


## Final Logistic

X_test = main.X_test
y_test = main.y_test
df_pca80 = dimred.df_pca80
lda4_df = dimred.lda4_df

df_pca80_test = dimred.df_pca80_test
lda4_df_test = dimred.lda4_df_test

#NIST-F: with outliers LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.01, penalty='l2')
#NIST_S: with outliers LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.01, penalty='l2')
#Sfing default: with outliers LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.1, penalty='l2')
#Sfing HQ: with outliers LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.01, penalty='l2')
#SfinGE_VQ: without outliers LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.01, penalty='l2')
import pandas as pd
clf = LogisticRegression(multi_class='multinomial', max_iter=1200,random_state=random_state, C=0.01, penalty='l2')

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

print('Logistic balanced accuracy score is: ', bal_acc_score)
print('precision_weighted is:', prec)
print('recall_weighted is:', recall)
print('F1 score is: ', F)
print('roc_auc_ovo_weighted is:', RAUC)
print('MCC is ', MCC)

"""
NIST-F Final DF PCA80:
balanced accuracy score is:  0.8006868686868687
precision_weighted is: 0.8497141731728337
recall_weighted is: 0.8545454545454545
F1 score is:  0.8453017514184911
roc_auc_ovo_weighted is: 0.9791029231711049
MCC is  0.8158165094530042

NIST-F final LDA4:
Logistic balanced accuracy score is:  0.6314702581369248
precision_weighted is: 0.6721291497793171
recall_weighted is: 0.6757575757575758
F1 score is:  0.6706806790617718
roc_auc_ovo_weighted is: 0.87996229976533
MCC is  0.5880942669037109

NIST-S final PCA80:
Logistic balanced accuracy score is:  0.8088215488215488
precision_weighted is: 0.8574512166284318
recall_weighted is: 0.8606060606060606
F1 score is:  0.8529783705935206
roc_auc_ovo_weighted is: 0.9742959391898786
MCC is  0.8234575653735846
NIST-S final lda4:
Logistic balanced accuracy score is:  0.6286060606060606
precision_weighted is: 0.6755162228227162
recall_weighted is: 0.6757575757575758
F1 score is:  0.6730990380559374
roc_auc_ovo_weighted is: 0.8849775278032853
MCC is  0.588696660553896

Sfing_Default PCA80:
Logistic balanced accuracy score is:  0.8971067189065522
precision_weighted is: 0.9565438481177926
recall_weighted is: 0.9565
F1 score is:  0.9564405953889195
roc_auc_ovo_weighted is: 0.9900051516055595
MCC is  0.9381066408505818

Sfing_Default LDA4:
Logistic balanced accuracy score is:  0.9039810986457436
precision_weighted is: 0.9604704165774256
recall_weighted is: 0.9595
F1 score is:  0.9599364465144468
roc_auc_ovo_weighted is: 0.9887603158709575
MCC is  0.9423992001520154

Sfing HQ PCA80:
Logistic balanced accuracy score is:  0.8969370556558989
precision_weighted is: 0.966812953433917
recall_weighted is: 0.9675
F1 score is:  0.9670540056049621
roc_auc_ovo_weighted is: 0.9930623054905723
MCC is  0.9537382705641269
sfing HQ LDA4:
Logistic balanced accuracy score is:  0.9058113375772583
precision_weighted is: 0.9659020134786229
recall_weighted is: 0.967
F1 score is:  0.9662322907811143
roc_auc_ovo_weighted is: 0.99177631063863
MCC is  0.9529774574163918

Sfing VQ PCA80:
Logistic balanced accuracy score is:  0.8864785175539793
precision_weighted is: 0.9600573008349848
recall_weighted is: 0.9617404351087772
F1 score is:  0.960242032388015
roc_auc_ovo_weighted is: 0.994947212257431
MCC is  0.9443968481089142
Sfing VQ LDA4:
Logistic balanced accuracy score is:  0.8541375705298615
precision_weighted is: 0.9585254870634329
recall_weighted is: 0.9609902475618904
F1 score is:  0.958831560642018
roc_auc_ovo_weighted is: 0.9928034372280274
MCC is  0.9433043439640263
"""