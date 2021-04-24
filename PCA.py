## Dimentionality reduction - PCA
## Irene Grone

## initial import
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA


## load data
pulsar = pd.read_csv('pulsar_stars.csv')


## renaming the features for readability
features_names = ["mean_integrated", "std_integrated", "exKurt_integrated", "skewness_integrated", "mean_DM-SNR", "std_DM-SNR", "exKurt_DM-SNR", "skewness_DM-SNR", "target"]
pulsar.columns = features_names


## pariplot of the original dataset
g = sns.pairplot(pulsar, hue="target")
plt.show()
input("press key to continue")


## separate features and target
X_pulsar = pulsar.iloc[:, :8]
y_pulsar = pulsar['target']
features_sc = ["mean_integrated", "std_integrated", "exKurt_integrated", "skewness_integrated", "mean_DM-SNR", "std_DM-SNR", "exKurt_DM-SNR", "skewness_DM-SNR"]


## scaling the dataset
X_scaled = preprocessing.scale(X_pulsar)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = features_sc


## pariplot of the scaled dataset
g2 = sns.pairplot(X_scaled)
plt.show()
input("press key to continue")


##3D plots for selecting the three predictors
fig = plt.figure(figsize=(30, 8))
ax = fig.add_subplot(1, 4, 1, projection='3d')
ax.scatter(X_scaled["exKurt_integrated"], X_scaled["skewness_DM-SNR"], X_scaled["exKurt_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax.set_title('Group 1', fontsize=14)
ax.set_xlabel("exKurt_integrated")
ax.set_ylabel("skewness_DM-SNR")
ax.set_zlabel("exKurt_DM-SNR")

ax = fig.add_subplot(1, 4, 2, projection='3d')
ax.scatter(X_scaled["mean_integrated"], X_scaled["exKurt_DM-SNR"], X_scaled["skewness_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax.set_title('Group 2', fontsize=14)
ax.set_xlabel("mean_integrated")
ax.set_ylabel("exKurt_DM-SNR")
ax.set_zlabel("skewness_DM-SNR")

ax = fig.add_subplot(1, 4, 3, projection='3d')
ax.scatter(X_scaled["mean_integrated"], X_scaled["mean_DM-SNR"], X_scaled["std_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax.set_title('Group 3', fontsize=14)
ax.set_xlabel("mean_integrated")
ax.set_ylabel("mean_DM-SNR")
ax.set_zlabel("std_DM-SNR")

ax = fig.add_subplot(1, 4, 4, projection='3d')
ax.scatter(X_scaled["skewness_integrated"], X_scaled["skewness_DM-SNR"], X_scaled["exKurt_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax.set_title('Group 4', fontsize=14)
ax.set_xlabel("skewness_integrated")
ax.set_ylabel("skewness_DM-SNR")
ax.set_zlabel("exKurt_DM-SNR")

plt.show()
input("press key to continue")


## selection of the predictor
features = X_scaled[["skewness_integrated", "skewness_DM-SNR", "exKurt_DM-SNR"]]


## applying PCA
pca = PCA(n_components = 2)
X_reduced = pca.fit_transform(features)


## print of components and variance ratio
print("PCA components")
print(pca.components_)
print("variance explained")
print(pca.explained_variance_ratio_)


## projection of dataset on the 3D space of selected predictors and on reduced space of PCs
fig2 = plt.figure(figsize=(16, 8))

ax2 = fig2.add_subplot(1, 2, 1, projection='3d')
ax2.scatter(X_scaled["skewness_integrated"], X_scaled["skewness_DM-SNR"], X_scaled["exKurt_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax2.set_title('3D space', fontsize=20)
ax2.set_xlabel("skewness_integrated", fontsize=14)
ax2.set_ylabel("skewness_DM-SNR", fontsize=14)
ax2.set_zlabel("exKurt_DM-SNR", fontsize=14)

ax2 = fig2.add_subplot(1, 2, 2)
ax2.scatter(X_scaled["skewness_integrated"], X_scaled["skewness_DM-SNR"], c=y_pulsar, cmap=cm.coolwarm)
ax2.set_title('Reduced space', fontsize=20)
ax2.set_xlabel("First PC", fontsize=14)
ax2.set_ylabel("Second PC", fontsize=14)

plt.show()
