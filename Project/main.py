from sklearn.model_selection import KFold, train_test_split # import KFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import IterativeImputer
import pandas as pd
import missingno as mno
import seaborn as sns
import numpy as np

load_path_NISTF = 'Project/data/NISTDB4-F.csv'
load_path_NISTS = 'Project/data/NISTDB4-S.csv'
load_path_SFING_Default = 'Project/data/SFinGe_Default.csv'
load_path_SFING_HQ = 'Project/data/SFinGe_HQNoPert.csv'
load_path_SFING_VQ = 'Project/data/SFinGe_VQAndPert.csv'
raw_data = pd.read_csv(load_path_NISTF, header=None)

random_state= 1

labels = raw_data.iloc[:,-1].to_frame()
inputs = raw_data.iloc[:,:-1]

#Split into train and test data, perform cross-validation on train data
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=random_state)

#Check counts of classes (balance)
sns.countplot(data=y_train, x=2014)

#First handle missing data
## Check which data is actually missing and how much data is missing
attributes_with_missing = np.sum(np.any(np.isnan(X_train), axis=0))
print('attributes with missing values:', attributes_with_missing)
missing_rate = np.count_nonzero(np.isnan(X_train))/np.prod(X_train.shape)
print('missing value rate', missing_rate)
# Calculate percentage of missing value per column within dataset
percent_missing = X_train.isnull().sum() * 100 / len(X_train)
missing_value_df = pd.DataFrame({'column_name': X_train.columns,
                                 'percent_missing': percent_missing})

## Imputing missing data through multiple imputation
imputer = IterativeImputer(random_state=random_state, verbose=2, max_iter=20)
imputed = imputer.fit_transform(X_train)

#Check correlations within features (high dimensional data for some datasets p>n)
corr_mat = X_train.corr()
mutual_info_classif(X=X_train, y=y_train)


#listwise deletion
X_train.dropna()


#k-fold cross-validation on train set
kf = KFold(n_splits=10,shuffle=True, random_state=random_state) # Define the split - into 2 folds
kf.get_n_splits(X_train) # returns the number of splitting iterations in the cross-validator
print(kf)


for train_index, test_index in kf.split(X_train):
 print('TRAIN:', train_index, 'TEST:', test_index)
