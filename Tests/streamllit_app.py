import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the data
@st.cache
def load_data():
    return pd.read_csv('/content/credit_card_customer_data.csv')

df = load_data()

# Select columns for analysis
cols_to_consider = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']
subset = df[cols_to_consider]

# Exploratory Data Analysis
st.subheader('Exploratory Data Analysis')
st.write(subset.head())

st.write('Missing Values:')
st.write(subset.isna().sum())

st.write('Summary Statistics:')
st.write(subset.describe())

# Standardize the data
scaler = StandardScaler()
subset_scaled = scaler.fit_transform(subset)
subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)

# Data Visualizations
st.subheader('Data Visualizations')

# Heatmap
st.write('Correlation Heatmap:')
st.write(sns.heatmap(subset_scaled_df.corr(), annot=True))
st.pyplot()

# Pairplot
st.write('Pairplot:')
st.write(sns.pairplot(subset_scaled_df, diag_kind="kde"))
st.pyplot()

# Elbow Curve
st.subheader('Elbow Curve')

clusters = range(1, 10)
meanDistortions = []

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(subset_scaled_df)
    prediction = model.predict(subset_scaled_df)
    distortion = sum(np.min(cdist(subset_scaled_df, model.cluster_centers_, 'euclidean'), axis=1)) / subset_scaled_df.shape[0]
    meanDistortions.append(distortion)
    st.write(k, distortion)

plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
st.pyplot()

# KMeans Clustering
st.subheader('KMeans Clustering')

kmeans = KMeans(n_clusters=3, n_init=15, random_state=2345)
kmeans.fit(subset_scaled_df)

centroids = kmeans.cluster_centers_
centroid_df = pd.DataFrame(centroids, columns=subset_scaled_df.columns)
st.write('Centroids:')
st.write(centroid_df)

# Adding Labels to the Dataset
dataset = subset_scaled_df.copy()
dataset['KmeansLabel'] = kmeans.labels_
st.write('Dataset with KMeans Labels:')
st.write(dataset.head())

# Visualizing the Clusters
st.write('Scatter plot - Avg_Credit_Limit vs Total_visits_online:')
plt.scatter(dataset['Avg_Credit_Limit'], dataset['Total_visits_online'], c=kmeans.labels_)
st.pyplot()

# Hierarchical Clustering
st.subheader('Hierarchical Clustering')

linkage_methods = ['single', 'complete', 'average', 'ward', 'median']
results_cophenetic_coef = []
for i in linkage_methods:
    plt.figure(figsize=(15, 13))
    plt.xlabel('sample index')
    plt.ylabel('Distance')
    Z = linkage(subset_scaled_df, i)
    cc, cophn_dist = cophenet(Z, pdist(subset_scaled_df))
    dendrogram(Z, leaf_rotation=90.0, p=5, leaf_font_size=10, truncate_mode='level')
    plt.tight_layout()
    plt.title("Linkage Type: " + i + " having cophenetic coefficient : " + str(round(cc, 3)))
    st.pyplot()
    results_cophenetic_coef.append((i, cc))
    st.write(i, cc)

results_cophenetic_coef_df = pd.DataFrame(results_cophenetic_coef, columns=['LinkageMethod', 'CopheneticCoefficient'])
st.write('Cophenetic Coefficients:')
st.write(results_cophenetic_coef_df)

# Selecting the number of clusters with Hierarchical Clustering
max_d = 5
clusters = fcluster(Z, max_d, criterion='distance')
st.write('Hierarchical Clustering Labels:')
st.write(set(clusters))

# Assign the clusters label to the dataset
dataset2 = subset_scaled_df.copy()
dataset2['HierarchicalClusteringLabel'] = clusters
st.write('Dataset with Hierarchical Clustering Labels:')
st.write(dataset2.head())

# Silhouette Score
from sklearn.metrics import silhouette_score
silhouette_score_kmeans = silhouette_score(dataset.drop('KmeansLabel', axis=1), dataset['KmeansLabel'])
silhouette_score_hierarchical = silhouette_score(dataset2.drop('HierarchicalClusteringLabel', axis=1), dataset2['HierarchicalClusteringLabel'])
st.write('Silhouette Score - KMeans Clustering:', silhouette_score_kmeans)
st.write('Silhouette Score - Hierarchical Clustering:', silhouette_score_hierarchical)

# Comparing KMeans and Hierarchical Results
st.subheader('Comparing KMeans and Hierarchical Results')

Kmeans_results = dataset.groupby('KmeansLabel').mean()
st.write('KMeans Results:')
st.write(Kmeans_results)

Hierarchical_results = dataset2.groupby('HierarchicalClusteringLabel').mean()
st.write('Hierarchical Results:')
st.write(Hierarchical_results)

# Cluster Profiles and Marketing Recommendations
st.subheader('Cluster Profiles and Marketing Recommendations')

subset['KmeansLabel'] = dataset['KmeansLabel']
for each in cols_to_consider:
    st.write(each)
    st.write(subset.groupby('KmeansLabel').describe().round()[each][['count', 'mean', 'min', 'max']])
    st.write("\n\n")
