import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import numpy as np

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

from scipy.cluster.hierarchy import fcluster

def main():
    st.title('ChannelSegmenter: Credit Card Costumer Inquiry Channel Segmentation App')

    # Load data
    @st.cache
    def load_data():
        return pd.read_csv('Tests/raw_data/credit_card_customer_data.csv')

    df = load_data()

    if st.checkbox("Show raw data"):
        st.write(df)

    cols_to_consider = ['Avg_Credit_Limit', 'Total_Credit_Cards', 'Total_visits_bank', 'Total_visits_online', 'Total_calls_made']
    subset = df[cols_to_consider]

    st.write("## Exploratory Data Analysis")
    st.write(subset.isna().sum())
    st.write(subset.describe())

    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)
    subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)

    st.write("## Data Visualization")

    # Checkbox to show/hide heatmap
    show_heatmap = st.checkbox("Show Heatmap")
    if show_heatmap:
        fig, ax = plt.subplots()
        sns.heatmap(subset_scaled_df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

    # Checkbox to show/hide pairplot
    show_pairplot = st.checkbox("Show Pairplot")
    if show_pairplot:
        pairplot = sns.pairplot(subset_scaled_df)
        st.pyplot(pairplot)

    st.write("Elbow Curve")
    clusters = range(1, 10)
    meanDistortions = []
    for k in clusters:
        model = KMeans(n_clusters=k)
        model.fit(subset_scaled_df)
        prediction = model.predict(subset_scaled_df)
        distortion = sum(np.min(cdist(subset_scaled_df, model.cluster_centers_, 'euclidean'), axis=1)) / subset_scaled_df.shape[0]
        meanDistortions.append(distortion)
    plt.figure(figsize=(8, 5))
    plt.plot(clusters, meanDistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    st.pyplot(plt)

    st.write("KMeans Clustering")
    kmeans = KMeans(n_clusters=3, n_init=15)
    kmeans.fit(subset_scaled_df)
    st.write(kmeans)

    st.write("Centroids for Kmeans")
    centroids = kmeans.cluster_centers_
    st.write(centroids)
    st.write("Centroids for DataFrame")
    centroid_df = pd.DataFrame(centroids, columns=subset_scaled_df.columns)
    st.write(centroid_df)

    dataset = subset_scaled_df[:]
    dataset['KmeansLabel'] = kmeans.labels_
    dataset.head(10)
    st.write(dataset)

    boxplot_fig, ax = plt.subplots()
    dataset.boxplot(by='KmeansLabel', layout=(2, 3), figsize=(20, 15), ax=ax)

    # Adjust the title position
    ax.set_title("Boxplot of KMeans Labels", pad=20)  # Set the title with padding

    # Adjust horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=2.5, hspace=0.25)

    # Display the plot in Streamlit
    st.pyplot(boxplot_fig)

    st.write("## Visualizing the Clusters")

    selected_variable_x = st.selectbox('Select X-axis variable', cols_to_consider[:-1])
    selected_variable_y = st.selectbox('Select Y-axis variable', cols_to_consider[:-1])

    scatter_fig, ax = plt.subplots()
    ax.scatter(dataset[selected_variable_x], dataset[selected_variable_y], c=kmeans.labels_, cmap='viridis')
    ax.set_xlabel(selected_variable_x)
    ax.set_ylabel(selected_variable_y)
    st.pyplot(scatter_fig)

    st.write("## Hierarchical Clustering")

    linkage_methods = ['single', 'complete', 'average', 'ward', 'median']
    selected_linkage = st.selectbox('Select linkage method', linkage_methods)

    plt.figure(figsize=(15, 13))
    plt.xlabel('sample index')
    plt.ylabel('Distance')
    Z = linkage(subset_scaled_df, selected_linkage)
    cc, cophn_dist = cophenet(Z, pdist(subset_scaled_df))
    dendrogram(Z, leaf_rotation=90.0, p=5, leaf_font_size=10, truncate_mode='level')
    plt.tight_layout()
    plt.title("Linkage Type: " + selected_linkage + " having cophenetic coefficient : " + str(round(cc, 3)))
    st.pyplot(plt)

    results_cophenetic_coef_df = pd.DataFrame([(selected_linkage, cc)], columns=['LinkageMethod', 'CopheneticCoefficient'])
    st.write(results_cophenetic_coef_df)

    plt.figure(figsize=(10, 8))
    Z = linkage(subset_scaled_df, 'complete', metric='euclidean')

    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=25  # show only the last p merged clusters
    )
    st.pyplot(plt)

    max_d = 5
    clusters = fcluster(Z, max_d, criterion='distance')
    set(clusters)

    dataset2 = subset_scaled_df[:]
    dataset2['HierarchicalClusteringLabel'] = clusters
    dataset2.head(3)
    st.write(dataset2)

    boxplot2_fig, ax = plt.subplots()
    dataset2.boxplot(by='HierarchicalClusteringLabel', layout=(2, 3), figsize=(20, 15), ax=ax)

    # Adjust the title position
    ax.set_title("Boxplot of KMeans Labels", pad=20)  # Set the title with padding

    # Adjust horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=2.5, hspace=0.25)

    # Display the plot in Streamlit
    st.pyplot(boxplot2_fig)

    # Dropdown interaction for silhouette score
    selected_label = st.selectbox('Select clustering label for silhouette score', ['KmeansLabel', 'HierarchicalClusteringLabel'])
    if selected_label == 'KmeansLabel':
        st.write("Silhouette Score for KmeansLabel")
        st.write(silhouette_score(dataset.drop('KmeansLabel', axis=1), dataset['KmeansLabel']))
    elif selected_label == 'HierarchicalClusteringLabel':
        st.write("Silhouette Score for HierarchicalClusteringLabel")
        st.write(silhouette_score(dataset2.drop('HierarchicalClusteringLabel', axis=1), dataset2['HierarchicalClusteringLabel']))

    st.write("## Comparing KMeans and Heirarchical Results")
    st.write("KMeans")
    Kmeans_results = dataset.groupby('KmeansLabel').mean()
    st.write(Kmeans_results)
    st.write("Heirarchical Results")
    Hierarchical_results = dataset2.groupby('HierarchicalClusteringLabel').mean()
    st.write(Hierarchical_results)
    st.write("KMeans")
    Kmeans_results.index = ['G1', 'G2', 'G3']
    st.write(Kmeans_results)
    st.write("Heirarchical Results")
    Hierarchical_results.index = ['G3', 'G1', 'G2']
    Hierarchical_results.sort_index(inplace=True)
    st.write(Hierarchical_results)

    Kmeans_results.plot.bar(figsize=(10, 6))
    plt.title('Mean Values of Clusters (KMeans)')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    Hierarchical_results.plot.bar(figsize=(10, 6))
    plt.title('Mean Values of Clusters (Hierarchical)')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=0)
    st.pyplot(plt)

    st.write("## Summary Statistics by Cluster")

# Define the number of columns for the layout
num_columns = 2

# Calculate the number of rows required
num_rows = len(cols_to_consider) // num_columns
if len(cols_to_consider) % num_columns != 0:
    num_rows += 1

# Loop through each variable and create a table for each
for i in range(num_rows):
    # Start a new row
    row = st.columns(num_columns)
    for j in range(num_columns):
        idx = i * num_columns + j
        if idx < len(cols_to_consider):
            variable = cols_to_consider[idx]
            table_data = subset.groupby('KmeansLabel').describe().round()[variable][['count', 'mean', 'min', 'max']]
            with row[j]:
                st.write("**{}**".format(variable))
                st.write(table_data)

    

if __name__ == "__main__":
    main()
