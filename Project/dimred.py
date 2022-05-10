import Project.main as main
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

random_state = main.random_state
mod_df = main.mod_df
y_train = main.y_train
# DIMENSIONALITY REDUCTION
## PCA


### Decide the number of PCA components based on the retained information

#pca = PCA(random_state=random_state)
#pca.fit(mod_df)
#explained_variance = np.cumsum(pca.explained_variance_ratio_)
#plt.vlines(x=404, ymax=1, ymin=0, colors="r", linestyles="--")
#plt.hlines(y=0.95, xmax=404, xmin=0, colors="g", linestyles="--")
# plt.plot(explained_variance)

### take 404 components that explain 95% of the variance and check correlation matrix of retained components
pca_80 = PCA(random_state=random_state, n_components=0.80)
pca_90 = PCA(random_state=random_state, n_components=0.90)
pca_95 = PCA(random_state=random_state, n_components=0.95)
pca_99 = PCA(random_state=random_state, n_components=0.99)

df_pca80 = pca_80.fit_transform(mod_df)
df_pca90 = pca_90.fit_transform(mod_df)
df_pca95 = pca_95.fit_transform(mod_df)
df_pca99 = pca_99.fit_transform(mod_df)
corr_mat_pca = np.corrcoef(df_pca95.transpose())
plt.figure(figsize=[15,8])
sns.heatmap(corr_mat_pca)
plt.show()

### Principal components are unrelated

## LDA
lda_df = LinearDiscriminantAnalysis(n_components=4).fit(X=mod_df, y=y_train.values.ravel()).transform(mod_df)

