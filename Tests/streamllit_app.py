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
from scipy.cluster.hierarchy import dendrogram, linkage,cophenet

from scipy.cluster.hierarchy import fcluster

def main():
    st.title('Customer Segmentation using Credit Card Data')

    # Load data
    @st.cache
    def load_data():
        return pd.read_csv('raw_data/credit_card_customer_data.csv')

    df = load_data()

    if st.checkbox("Show raw data"):
        st.write(df)

    cols_to_consider=['Avg_Credit_Limit','Total_Credit_Cards','Total_visits_bank','Total_visits_online','Total_calls_made']
    subset=df[cols_to_consider]

    st.write("## Exploratory Data Analysis")
    st.write(subset.isna().sum())
    st.write(subset.describe())

    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)
    subset_scaled_df = pd.DataFrame(subset_scaled, columns=subset.columns)

    st.write("## Data Visualization")

    st.write("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(subset_scaled_df.corr(), annot=True, ax=ax)
    st.pyplot(fig)

    st.write("Pairplot")
    pairplot = sns.pairplot(subset_scaled_df)
    st.pyplot(pairplot)

    st.write("Elbow Curve")
    clusters=range(1,10)
    meanDistortions=[]
    for k in clusters:
        model=KMeans(n_clusters=k)
        model.fit(subset_scaled_df)
        prediction=model.predict(subset_scaled_df)
        distortion=sum(np.min(cdist(subset_scaled_df, model.cluster_centers_, 'euclidean'), axis=1)) / subset_scaled_df.shape[0]
        meanDistortions.append(distortion)
    plt.figure(figsize=(8,5))
    plt.plot(clusters, meanDistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    st.pyplot(plt)

    st.write("KMeans Clustering")
    kmeans = KMeans(n_clusters=3, n_init = 15)
    kmeans.fit(subset_scaled_df)
    st.write(kmeans)

    st.write("Centroids for Kmeans")
    centroids = kmeans.cluster_centers_
    st.write(centroids)
    st.write("Centroids for DataFrame")
    centroid_df = pd.DataFrame(centroids, columns = subset_scaled_df.columns )
    st.write(centroid_df)

    dataset=subset_scaled_df[:]
    dataset['KmeansLabel']=kmeans.labels_
    dataset.head(10)
    st.write(dataset)

    boxplot_fig, ax = plt.subplots()
    dataset.boxplot(by='KmeansLabel', layout=(2,3), figsize=(20, 15), ax=ax)

    # Adjust the title position
    ax.set_title("Boxplot of KMeans Labels", pad=20)  # Set the title with padding

    # Adjust horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=2.5, hspace=0.25)

    # Display the plot in Streamlit
    st.pyplot(boxplot_fig)

    st.write("## Visualizing the Clusters")

    scatter_fig, ax = plt.subplots()
    ax.scatter(dataset['Avg_Credit_Limit'], dataset['Total_visits_online'], c=kmeans.labels_, cmap='viridis')
    ax.set_xlabel('Avg_Credit_Limit')
    ax.set_ylabel('Total_visits_online')
    st.pyplot(scatter_fig)

    scatter_fig, ax = plt.subplots()
    ax.scatter(dataset['Avg_Credit_Limit'], dataset['Total_visits_bank'], c=kmeans.labels_, cmap='viridis')
    ax.set_xlabel('Avg_Credit_Limit')
    ax.set_ylabel('Total_visits_bank')
    st.pyplot(scatter_fig)

    scatter_fig, ax = plt.subplots()
    ax.scatter(dataset['Avg_Credit_Limit'], dataset['Total_calls_made'], c=kmeans.labels_, cmap='viridis')
    ax.set_xlabel('Avg_Credit_Limit')
    ax.set_ylabel('Total_calls_made')
    st.pyplot(scatter_fig)

    st.write("## Hierarchical Clustering")

    linkage_methods=['single','complete','average','ward','median']
    results_cophenetic_coef=[]
    for i in linkage_methods :
        plt.figure(figsize=(15, 13))
        plt.xlabel('sample index')
        plt.ylabel('Distance')
        Z = linkage(subset_scaled_df, i)
        cc,cophn_dist=cophenet(Z,pdist(subset_scaled_df))
        dendrogram(Z,leaf_rotation=90.0,p=5,leaf_font_size=10,truncate_mode='level')
        plt.tight_layout()
        plt.title("Linkage Type: "+ i +" having cophenetic coefficient : "+str(round(cc,3)) )
        st.pyplot(plt)
        results_cophenetic_coef.append((i,cc))
        print (i,cc)

    results_cophenetic_coef_df=pd.DataFrame(results_cophenetic_coef,columns=['LinkageMethod','CopheneticCoefficient'])
    st.write(results_cophenetic_coef_df)

    plt.figure(figsize=(10,8))
    Z = linkage(subset_scaled_df, 'complete', metric='euclidean')

    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=25 # show only the last p merged clusters
    )
    st.pyplot(plt)

    max_d=5
    clusters = fcluster(Z, max_d, criterion='distance')
    set(clusters)

    dataset2=subset_scaled_df[:]
    dataset2['HierarchicalClusteringLabel']=clusters
    dataset2.head(3)
    st.write(dataset2)

    boxplot2_fig, ax = plt.subplots()
    dataset2.boxplot(by='HierarchicalClusteringLabel', layout=(2,3), figsize=(20, 15), ax=ax)

    # Adjust the title position
    ax.set_title("Boxplot of KMeans Labels", pad=20)  # Set the title with padding

    # Adjust horizontal and vertical spacing between subplots
    plt.subplots_adjust(wspace=2.5, hspace=0.25)

    # Display the plot in Streamlit
    st.pyplot(boxplot2_fig)

    st.write("Silhouette Score for KmeansLabel")
    st.write(silhouette_score(dataset.drop('KmeansLabel',axis=1),dataset['KmeansLabel']))
    st.write("Silhouette Score for HierarchicalClusteringLabel")
    st.write(silhouette_score(dataset2.drop('HierarchicalClusteringLabel',axis=1),dataset2['HierarchicalClusteringLabel']))

    st.write("## Comparing KMeans and Heirarchical Results")
    st.write("KMeans")
    Kmeans_results=dataset.groupby('KmeansLabel').mean()
    st.write(Kmeans_results)
    st.write("Heirarchical Results")
    Hierarchical_results=dataset2.groupby('HierarchicalClusteringLabel').mean()
    st.write(Hierarchical_results)
    st.write("KMeans")
    Kmeans_results.index=['G1','G2','G3']
    st.write(Kmeans_results)
    st.write("Heirarchical Results")
    Hierarchical_results.index=['G3','G1','G2']
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

    subset['KmeansLabel']=dataset['KmeansLabel']
    for each in cols_to_consider:
        st.write(each)
        st.write(subset.groupby('KmeansLabel').describe().round()[each][['count','mean','min','max']])
        st.write("\n\n")

if __name__ == "__main__":
    main()
