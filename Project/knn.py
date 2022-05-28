from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, matthews_corrcoef
import numpy as np
import Project.main as main
import Project.dimred as dimred
from Project.classif_automation import classif_automation
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

random_state = main.random_state
y_train= main.y_train
#create a new KNN model
knn_cv = KNeighborsClassifier()
print('start:', datetime.now())
results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10]
default_neighbors = [5]
mod_df = main.mod_df
scoring_methods = ['balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                       'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]

for scoring in scoring_methods:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=5))])
    # evaluate the model
    strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring=scoring, cv=strat_kfolds, n_jobs=-1)
    # store results
    results.append(n_scores)
    print(scoring, np.mean(n_scores), np.std(n_scores))
print('end base:', datetime.now())
# n = 3 or 5 gives best balanced accuracy

print('start pca:', datetime.now())
## Scores for pca dim reduction
df_pca99 = dimred.df_pca99
df_pca90 = dimred.df_pca90
df_pca95 = dimred.df_pca95
df_pca80 = dimred.df_pca80
results = list()
neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]

datasets_pca = [df_pca80, df_pca90, df_pca95, df_pca99]
for dataset in datasets_pca:
    for n in default_neighbors:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
        # evaluate the model
        strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(pipeline, dataset, y_train.values.ravel(),
                                   scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
        # store results
        results.append(n_scores)
        print(len(dataset[0]), n, np.mean(n_scores), np.std(n_scores))
classif_automation(datasets=datasets_pca, classifier=KNeighborsClassifier(n_neighbors=5),
                   y_train=y_train,
                   random_state=random_state)

print('end pca:', datetime.now())
## LDA

print('start lda:', datetime.now())
results = list()
neighbors = [1, 2, 3, 4,5, 6, 7, 8, 9, 10, 20, 50, 100]
lda4_df = dimred.lda4_df
lda3_df = dimred.lda3_df
lda2_df = dimred.lda2_df
lda1_df = dimred.lda1_df
datasets_lda= [lda4_df, lda3_df, lda2_df, lda1_df]


classif_automation(datasets=datasets_lda, classifier=KNeighborsClassifier(n_neighbors=5),
                   y_train=y_train,
                   random_state=random_state)

for scoring in scoring_methods:
    for n in default_neighbors:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
        # evaluate the model
        strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=main.random_state)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(pipeline, lda_df, y_train.values.ravel(), scoring=scoring, cv=strat_kfolds, n_jobs=-1)
        # store results
        results.append(n_scores)
        print('lda',len(lda_df[0]), n, np.mean(n_scores), np.std(n_scores))


##ISOMAP df
print('start iso:', datetime.now())
iso_df = dimred.iso_df
datasets_iso = [iso_df]
classif_automation(datasets=datasets_iso, classifier=KNeighborsClassifier(n_neighbors=5),
                   y_train=y_train,
                   random_state=random_state)
print('end iso:', datetime.now())

##hyperparam optimization

print('start hyperparam knn optimization:', datetime.now())

neighbors = list(range(1,31))
default_neighbors = [5]
mod_df = main.mod_df
df_pca80 = dimred.df_pca80
scoring_methods = ['balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                       'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
balanced_acc = []
for n in neighbors:
    print('\n start scoring for neighbors=', n, 'at', datetime.now(), '\n')
    for scoring in scoring_methods:
        # create the modeling pipeline
        pipeline = Pipeline(steps=[('m',KNeighborsClassifier(n_neighbors=n))])
        # evaluate the model
        strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        # evaluate the model and collect the scores
        n_scores = cross_val_score(pipeline, df_pca80, y_train.values.ravel(), scoring=scoring, cv=strat_kfolds, n_jobs=-1)
        # store results
        results.append(n_scores)
        print(n, scoring, np.mean(n_scores), np.std(n_scores))
        if scoring == 'balanced_accuracy':
            balanced_acc.append(np.mean(n_scores))
data_chosen = main.data_chosen
plt.clf()
sns.lineplot(y=balanced_acc, x=neighbors , marker='o').set_xticks(range(1,31, 2))
plt.ylabel('balanced_accuracy')
plt.axvline(x=balanced_acc.index(max(balanced_acc))+1)
plt.title(str(data_chosen.split('/')[-1]))
plt.show()

print('end knn hyperparam:', datetime.now())


## grid search comments
"""
Optimal n for neighbours:

NIST-F with outliers:
Top 3: 1) k=9 balanced_accuracy 0.755869238582089 2)k=7 balanced_accuracy 0.7554142177230491 3) k=5 balanced_accuracy 0.754177146550494
NIST-F without outliers:
Top 3: 1) k=5 balanced_accuracy 0.7542241865476744 2) k=7 balanced_accuracy 0.7507208067897914 3) k=1 balanced_accuracy 0.7470022351193629

NIST-S with outliers:
Top 3: 1) k=11 balanced_accuracy 0.7447980492390858 2) k=3 balanced_accuracy 0.7422762458764575 3) k=12 balanced_accuracy 0.7394118002568557
NIST-S without outliers:
Top 3: 1) k=1 balanced_accuracy 0.7256915794689842 2) k= 3 balanced_accuracy 0.7209039075605691 3)k=5 balanced_accuracy 0.7149383306344004

Sfinge_default with outliers:
Top 3: 1) k=1 balanced_accuracy 0.8603471434884205 2)k=3 balanced_accuracy 0.8516752422207418 3)k=7 balanced_accuracy 0.8366360267108339
Sfinge_default without outliers:
Top 3: 1) k=5 balanced_accuracy 0.8352044088292084 2) k=3 balanced_accuracy 0.8335520357019881 3)k=7 balanced_accuracy 0.831581630763844

Sfinge HQ with outliers:
top 3: 1)k=1 balanced_accuracy 0.8706316894395997 2) k=3 balanced_accuracy 0.8684367065300176 3)k=5 balanced_accuracy 0.8562269646115983
Sfinge HQ without outliers:
top 3: 1) k=1 0.8768400633580656 2) k=3 0.8467352284021363 3) k=9 balanced_accuracy 0.8366166241014117

Sfinge_VQ with outliers:
top 3: 1) k=1 balanced_accuracy 0.8123041337960236 2)k=3 balanced_accuracy 0.8030736018678647 3) k=7
Sfinge_VQ without outliers:
top 3: 1) k=1 balanced_accuracy 0.8077600423354468 2)k=3 balanced_accuracy 0.7948057670835396 3) k=7 balanced_accuracy 0.7849539367776339
"""


## Final KNN

X_test = main.X_test
y_test = main.y_test
df_pca80 = dimred.df_pca80
lda4_df = dimred.lda4_df

df_pca80_test = dimred.df_pca80_test
lda4_df_test = dimred.lda4_df_test

#NIST-F: with outliers KNeighborsClassifier(n_neighbors=9)
#NIST_S: with outliers KNeighborsClassifier(n_neighbors=11)
#Sfing default: with outliers KNeighborsClassifier(n_neighbors=3)
#Sfing HQ: with outliers KNeighborsClassifier(n_neighbors=3)
#SfinGE_VQ: with outliers KNeighborsClassifier(n_neighbors=3)
import pandas as pd
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(df_pca80, y_train.values.ravel())
pred_clf = clf.predict(df_pca80_test)
y_pred_proba=pd.DataFrame(clf.predict_proba(df_pca80_test))


from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

bal_acc_score = balanced_accuracy_score(y_pred=pred_clf, y_true=y_test)
F = f1_score(y_pred=pred_clf, y_true=y_test, average='weighted')
prec = precision_score(y_pred=pred_clf, y_true=y_test, average='weighted')
recall = recall_score(y_pred=pred_clf, y_true=y_test, average='weighted')
RAUC = roc_auc_score(y_true=y_test[y_test.columns[-1]], y_score=y_pred_proba ,multi_class='ovo', average='weighted')
MCC = matthews_corrcoef(y_pred=pred_clf, y_true=y_test)

print('KNN balanced accuracy score is: ', bal_acc_score)
print('precision_weighted is:', prec)
print('recall_weighted is:', recall)
print('F1 score is: ', F)
print('roc_auc_ovo_weighted is:', RAUC)
print('MCC is ', MCC)
"""
NIST-F final DF_PCA80
KNN balanced accuracy score is:  0.7820561167227834
precision_weighted is: 0.8472312409812409
recall_weighted is: 0.8272727272727273
F1 score is:  0.820492300221267
roc_auc_ovo_weighted is: 0.9531671087984218
MCC is  0.7863870779317991

NIST-F final LDA_4:
KNN balanced accuracy score is:  0.6381503928170595
precision_weighted is: 0.6771153941799768
recall_weighted is: 0.6818181818181818
F1 score is:  0.6771901556732574
roc_auc_ovo_weighted is: 0.8166778389960208
MCC is  0.595050994649356

NIST-S final DF_PCA80:
KNN balanced accuracy score is:  0.7167676767676767
precision_weighted is: 0.8268579072080594
recall_weighted is: 0.7909090909090909
F1 score is:  0.7703093265728037
roc_auc_ovo_weighted is: 0.954383638574295
MCC is  0.7453470231985067

NIST-S final LDA_4:
KNN balanced accuracy score is:  0.651331088664422
precision_weighted is: 0.6972617475828692
recall_weighted is: 0.6909090909090909
F1 score is:  0.6927654431626916
roc_auc_ovo_weighted is: 0.8211467452300786
MCC is  0.6082946861189148

Sfinge_default final PCA80 k=3 :
KNN balanced accuracy score is:  0.8441255021734207
precision_weighted is: 0.93829470960385
recall_weighted is: 0.9405
F1 score is:  0.9378262676333888
roc_auc_ovo_weighted is: 0.9646780399008704
MCC is  0.9156325407592746

sfinge default final LDA_4:
KNN balanced accuracy score is:  0.8895088029173394
precision_weighted is: 0.9548826376591972
recall_weighted is: 0.954
F1 score is:  0.9544161565073235
roc_auc_ovo_weighted is: 0.9645542396725042
MCC is  0.9345798322165725

sfinge_HQ final pca80:
KNN balanced accuracy score is:  0.8462970500559429
precision_weighted is: 0.9469554781404
recall_weighted is: 0.949
F1 score is:  0.9457095660063409
roc_auc_ovo_weighted is: 0.9776611444053473
MCC is  0.9278700063914511

sfinge HQ final lda_4:
KNN balanced accuracy score is:  0.9101470553092146
precision_weighted is: 0.9684196655827576
recall_weighted is: 0.9685
F1 score is:  0.9684290147580039
roc_auc_ovo_weighted is: 0.9635286655930307
MCC is  0.9551870202980765

sfingeVQ final pca80:
KNN balanced accuracy score is:  0.827095561961564
precision_weighted is: 0.9132647017667138
recall_weighted is: 0.913
F1 score is:  0.9101477489716194
roc_auc_ovo_weighted is: 0.9533544643938109
MCC is  0.8771694720033023

sfingeVQ final LDA4:
KNN balanced accuracy score is:  0.8475634533811061
precision_weighted is: 0.9303714743357115
recall_weighted is: 0.9315
F1 score is:  0.9297878428978452
roc_auc_ovo_weighted is: 0.9473604430154847
MCC is  0.9023442495985032

"""