import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = pd.read_csv("Iris.csv")
df.drop('Id', axis=1, inplace=True) # klasterləşmə və ya model qurmaq üçün faydalı məlumat daşımır

le = LabelEncoder()
df["Species_encoded"] = le.fit_transform(df["Species"])

# X dəyişənlərini seç və miqyasla
X = df.drop(['Species', 'Species_encoded'], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering

kmeans = KMeans(n_clusters=3, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

# PCA ilə 2D vizualizasiya
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='kmeans_cluster', palette='Set1', s=100)
plt.title("K-Means Klasterləşmə (PCA ilə)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid(True)
plt.show()

# Hierarchical Clustering

linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, labels=df["Species"].values, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Nümunələr")
plt.ylabel("Məsafə")
plt.tight_layout()
plt.show()

df['hierarchical_cluster'] = fcluster(linkage_matrix, 3, criterion='maxclust') - 1

#  PCA ilə vizualizasiya
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='hierarchical_cluster', palette='Set2', s=100)
plt.title("Hierarchical Clustering (PCA ilə)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid(True)
plt.show()

#  Real növlərlə müqayisə

print("\nSpecies ilə KMeans klasterlərinin kəsişməsi:")
print(pd.crosstab(df['Species'], df['kmeans_cluster']))

print("\nSpecies ilə Hierarchical klasterlərinin kəsişməsi:")
print(pd.crosstab(df['Species'], df['hierarchical_cluster']))