from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold # import KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import missingno as mno
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


load_path_NISTF = '/Users/parcifalvangucht/PycharmProjects/BDS_data/NISTDB4-F.csv'
load_path_NISTS = '/Users/parcifalvangucht/PycharmProjects/BDS_data/NISTDB4-S.csv'
load_path_SFING_Default = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_Default.csv'
load_path_SFING_HQ = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_HQNoPert.csv'
load_path_SFING_VQ = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_VQAndPert.csv'
raw_data = pd.read_csv(load_path_NISTF, header=None)

random_state= 1

labels = raw_data.iloc[:,-1].to_frame()
inputs = raw_data.iloc[:,:-1]

#Split into train and test data, perform cross-validation on train data
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=random_state)

#Check counts of classes (balance)
class_countsplot = sns.countplot(data=y_train, x=2014)

#First handle missing data
## Check which data is actually missing and how much data is missing
attributes_with_missing = np.sum(np.any(np.isnan(X_train), axis=0))
print('attributes with missing values starting point:', attributes_with_missing)
missing_rate = np.count_nonzero(np.isnan(X_train))/np.prod(X_train.shape)
print('missing value rate at start', missing_rate)
# Calculate percentage of missing value per column within dataset
percent_missing = X_train.isnull().sum() * 100 / len(X_train)
missing_value_df = pd.DataFrame({'column_name': X_train.columns,
                                 'percent_missing': percent_missing}).sort_values(by=['percent_missing'], ascending=False)

#Check correlations within features (high dimensional data for some datasets p>n)
corr_mat = X_train.corr()
upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape),k=1).astype(bool))
## columns that contain high correlations
to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column].abs() > 0.95)]
high_corr_mat = corr_mat[to_drop_corr]
print(to_drop_corr)
# Check high correlating variables that also have more than 40% missing
to_drop_missing = missing_value_df[(missing_value_df['percent_missing'] > 40)]
to_drop_both = to_drop_missing[to_drop_missing['column_name'].isin(to_drop_corr)]
### No high correlating variables that also have more than 40% missing

## Columns with more than 50% missing data will be deleted
perc = 40.0
min_count = int(((100-perc)/100)*X_train.shape[0] + 1)
mod_df = X_train.dropna( axis=1,
                thresh=min_count)

missing_rate_mod = np.count_nonzero(np.isnan(mod_df))/np.prod(mod_df.shape)
print('missing value rate after deletion of missing', missing_rate_mod)

## Columns with high correlation features

## Testing SimpleImputation Strategies with logistic regression: all .88, except

#results = list()
#strategies = ['mean', 'median', 'most_frequent', 'constant']

##SCALE DATA
scaler = StandardScaler()
mod_df = scaler.fit_transform(mod_df)
#for s in strategies:
    # create the modeling pipeline
    # pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', LogisticRegression(multi_class='multinomial', max_iter=500, random_state=random_state))])
    # evaluate the model
    # strat_kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    # evaluate the model and collect the scores
    # n_scores = cross_val_score(pipeline, mod_df, y_train.values.ravel(), scoring='balanced_accuracy', cv=strat_kfolds, n_jobs=-1)
    # store results
    # results.append(n_scores)
    # print(s, np.mean(n_scores), np.std(n_scores))

## Median is chosen as SimpleImputation strategy

## Imputing missing data through multiple imputation: NOT POSSIBLE ==> LARGE COMPUTING TIME
#imputer = IterativeImputer(random_state=random_state, verbose=2, max_iter=20)
#imputed = imputer.fit_transform(mod_df)

## IMPUTE DATA
imputer = SimpleImputer(strategy='median')
mod_df = imputer.fit_transform(mod_df)

missing_rate_mod = np.count_nonzero(np.isnan(mod_df))/np.prod(mod_df.shape)
print('missing value rate after amputation', missing_rate_mod)
