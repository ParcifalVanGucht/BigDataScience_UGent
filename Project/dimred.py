import Project.main as main
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime

random_state = main.random_state
mod_df = main.mod_df
y_train = main.y_train
X_test = main.X_test
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
df_pca80_test = pca_80.transform(X_test)

df_pca90 = pca_90.fit_transform(mod_df)
df_pca95 = pca_95.fit_transform(mod_df)
df_pca99 = pca_99.fit_transform(mod_df)
#corr_mat_pca = np.corrcoef(df_pca95.transpose())
#plt.figure(figsize=[15,8])
#sns.heatmap(corr_mat_pca)
#plt.show()

### Principal components are unrelated

## LDA
#max_n_components=4
print('start creation lda df:', datetime.now())
lda_4 = LinearDiscriminantAnalysis(n_components=4)
lda4_df = lda_4.fit_transform(X=mod_df, y=y_train.values.ravel())
lda4_df_test = lda_4.transform(X=X_test)

lda_3 = LinearDiscriminantAnalysis(n_components=3)
lda3_df = lda_3.fit_transform(X=mod_df, y=y_train.values.ravel())

lda_2 = LinearDiscriminantAnalysis(n_components=2)
lda2_df = lda_2.fit_transform(X=mod_df, y=y_train.values.ravel())

lda_1 = LinearDiscriminantAnalysis(n_components=1)
lda1_df = lda_1.fit_transform(X=mod_df, y=y_train.values.ravel())
print('end creation lda df:', datetime.now())

"""
dist_manhattan = manhattan_distances(mod_df)

## Multi-dimensional scaling (non-linear supervised dimensionality reduction)
stress = []
#dist_euclid = euclidean_distances(mod_df)
#euclidian distances not worth that much in high dimensional space
#manhattan distances perform better

# Max value for n_components
max_range = 20
print('start checking for optimal dim mds:', datetime.now())
for dim in range(1, max_range):
    print(dim, datetime.now())
    # Set up the MDS object
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=random_state, n_jobs=-1)
    # Apply MDS
    pts = mds.fit_transform(dist_manhattan)
    # Retrieve the stress value
    stress.append(mds.stress_)
# Plot stress vs. n_components
plt.clf()
plt.plot(range(1, max_range), stress)
plt.bar(range(1, max_range), stress)
plt.xticks(range(1, max_range, 1))
plt.xlabel('n_components')
plt.ylabel('stress')
plt.show()


final_mds = MDS(n_components=5, dissimilarity='precomputed', random_state=random_state)
mds_df = final_mds.fit_transform(dist_manhattan)
print('end checking for optimal dim mds:', datetime.now())
"""

print('start iso creation',datetime.now())
iso = Isomap().fit(mod_df)
iso_df = iso.transform(mod_df)
print('end iso creation',datetime.now())





