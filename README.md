Clustering using K-Means and Hierarchical Clustering

Clustering Analysis on the Iris Dataset
In this project, we analyze the classic Iris dataset using two different clustering methods:
✅ K-Means Clustering
✅ Hierarchical Clustering

📁 Files
Iris.csv — The classic Iris dataset

Iris_classification.py — Python script performing the analysis

📊 Methods and Algorithms Used
🔹 Data Preprocessing
The Id column was removed, as it does not contain useful information for clustering.

The Species column was encoded into numerical values using LabelEncoder.

Features used for clustering (X) were scaled using StandardScaler.

🔹 K-Means Clustering
KMeans was applied with n_clusters=3.

Results were visualized in 2D using PCA dimensionality reduction.

🔹 Hierarchical Clustering
A linkage matrix was created using the ward method.

A dendrogram was plotted to visualize sample distances.

Clusters were formed using fcluster, and PCA was used again for 2D visualization.

🧪 Evaluation
The clustering results were compared with the actual Species labels using:

print(pd.crosstab(df['Species'], df['kmeans_cluster']))
print(pd.crosstab(df['Species'], df['hierarchical_cluster']))
This allows us to evaluate how closely the clustering matches the true class labels.

🛠️ Libraries Used
pandas
numpy
matplotlib
seaborn
sklearn
scipy

📚 Source
Iris Dataset – UCI Machine Learning Repository
