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
from Project.helpers import detect_outlying_inds_by_iqr


load_path_NISTF = '/Users/parcifalvangucht/PycharmProjects/BDS_data/NISTDB4-F.csv'
load_path_NISTS = '/Users/parcifalvangucht/PycharmProjects/BDS_data/NISTDB4-S.csv'
load_path_SFING_Default = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_Default.csv'
load_path_SFING_HQ = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_HQNoPert.csv'
load_path_SFING_VQ = '/Users/parcifalvangucht/PycharmProjects/BDS_data/SFinGe_VQAndPert.csv'
data_chosen = load_path_SFING_VQ
raw_data = pd.read_csv(data_chosen, header=None)

random_state= 1

labels = raw_data.iloc[:,-1].to_frame()
inputs = raw_data.iloc[:,:-1]

#Split into train and test data, perform cross-validation on train data
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=random_state)

#Check counts of classes (balance)
number_inst = len(y_train)
class_subplot=sns.countplot(data=y_train, x=y_train.values.ravel())

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

## Columns with more than 40% missing data will be deleted
perc = 40.0
min_count = int(((100-perc)/100)*X_train.shape[0] + 1)
mod_df = X_train.dropna(axis=1,
                thresh=min_count)
min_count_test = int(((100-perc)/100)*X_test.shape[0] + 1)
X_test = X_test[mod_df.columns]

missing_rate_mod = np.count_nonzero(np.isnan(mod_df))/np.prod(mod_df.shape)
print('missing value rate within training set after deletion of missing', missing_rate_mod)

missing_rate_mod_test = np.count_nonzero(np.isnan(X_test))/np.prod(X_test.shape)
print('missing value rate within test set after deletion of missing', missing_rate_mod_test)


## Outlier detection training set mod_df: first IQR
outliers_df = {}
tmp = {}
for col in pd.DataFrame(mod_df):
    tmp[col]= detect_outlying_inds_by_iqr(pd.DataFrame(mod_df)[col])
    outliers_in_col = detect_outlying_inds_by_iqr(pd.DataFrame(mod_df)[col])
    #count times that a row is an outlier and create dict of counts
    for row in outliers_in_col:
        label = mod_df.index[row]
        outliers_df[label] = outliers_df.get(label, 0) + 1
counter =0
ratio = 0.1
for key, value in outliers_df.items():
    if value > len(mod_df.columns)*ratio:
        counter += 1
        print(key, value/len(mod_df.columns))
        mod_df.drop(key, axis=0, inplace=True)
        y_train.drop(key, axis=0, inplace=True)
print('Amount of rows that are outliers for more then', ratio, 'of the columns of the training set:',counter)

##Outlier detection test set
outliers_df = {}
tmp = {}
for col in pd.DataFrame(X_test):
    tmp[col]= detect_outlying_inds_by_iqr(pd.DataFrame(X_test)[col])
    outliers_in_col = detect_outlying_inds_by_iqr(pd.DataFrame(X_test)[col])
    #count times that a row is an outlier and create dict of counts
    for row in outliers_in_col:
        label = X_test.index[row]
        outliers_df[label] = outliers_df.get(label, 0) + 1
counter =0
ratio = 0.1
for key, value in outliers_df.items():
    if value > len(mod_df.columns)*ratio:
        counter += 1
        print(key, value/len(X_test.columns))
        X_test.drop(key, axis=0, inplace=True)
        y_test.drop(key, axis=0, inplace=True)
print('Amount of rows that are outliers for more then', ratio, 'of the columns of the test set:',counter)

## Columns with high correlation features

## Testing SimpleImputation Strategies with logistic regression: all .88, except

#results = list()
#strategies = ['mean', 'median', 'most_frequent', 'constant']


##SCALE DATA
scaler = StandardScaler()
mod_df = scaler.fit_transform(mod_df)
X_test = scaler.transform(X_test)
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
X_test = imputer.transform(X_test)

missing_rate_mod = np.count_nonzero(np.isnan(mod_df))/np.prod(mod_df.shape)
print('missing value rate after imputation', missing_rate_mod)


"""
def mahalanobis(x=None, data=None, cov=None):
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()

df_mah = pd.DataFrame()
input_mah = pd.DataFrame(mod_df)
#create new column in dataframe that contains Mahalanobis distance for each row
from scipy.stats import chi2
df_mah['mahalanobis'] = mahalanobis(x=input_mah, data=input_mah[1:2014])
df_mah['p_value'] = 1 - chi2.cdf(df_mah['mahalanobis'], 2)

# Extreme values with a significance level of 0.01
df_mah.loc[df_mah.p_value < 0.01].head(10)
import numpy as np

def calculate_mahalanobis_distance(y=None, data=None, cov=None):
    y_mu = y - np.mean(data, axis = 0)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def calculate_mahalanobis_distance_iteratevely(y_train, X_train):
    mahal_distances = []
    mahal_distances = np.empty(shape=[len(y_train)])
    step = 1000
    for left_bound in range(0, len(y_train), step):
        print(f"Mahal Distance {100 * round(left_bound / len(y_train), 2)}% done")
        right_bound = len(y_train) + 1 if left_bound + step > len(y_train) else left_bound + step
        # put the mahal_distances in the numpy array
        mahal_distances[left_bound:right_bound] = calculate_mahalanobis_distance(y=X_train[left_bound:right_bound],
                                                                                 data=X_train)
    return mahal_distances

df_mah_iter = pd.DataFrame()
df_mah_iter['mahalanobis'] = calculate_mahalanobis_distance_iteratevely(input_mah, input_mah)
df_mah_iter['p_value'] = 1 - chi2.cdf(df_mah_iter['mahalanobis'], 2)

# Extreme values with a significance level of 0.01
df_mah_iter.loc[df_mah_iter.p_value < 0.01].head(10)
"""

