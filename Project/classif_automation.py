def classif_automation(datasets, classifier, y_train, random_state):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, matthews_corrcoef
    from datetime import datetime
    import numpy as np
    results = list()
    scoring_methods = ['balanced_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                       'roc_auc_ovo_weighted', make_scorer(matthews_corrcoef)]
    start = datetime.now()
    print(classifier, 'start:', start)

    for dataset in datasets:
        print('nfeatures | scoring| mean | std \n')
        for scoring_m in scoring_methods:
            # create the modeling pipeline
            pipeline = Pipeline(steps=[('m', classifier)])
            # evaluate the model
            strat_kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            # evaluate the model and collect the scores
            n_scores = cross_val_score(pipeline, dataset, y_train.values.ravel(), scoring=scoring_m, cv=strat_kfolds,
                                       n_jobs=-1)
            # store results
            results.append([np.mean(n_scores),np.std(n_scores)])
            print(len(dataset[0]), scoring_m, np.mean(n_scores), np.std(n_scores))
    end = datetime.now()
    print('runtime:',end-start)
    return results