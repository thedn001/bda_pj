# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 20:02:51 2025

@author: Nitro5
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np


# load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
#set the page config
st.set_page_config(page_title= "K-Means Clustering", layout = "centered")    

# set the title
st.title("K-Means Clustering Visualizer")

# Sidebar slider to select k
st.sidebar.markdown("Configure Clustering")
k = st.sidebar.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

# Load Iris dataset
iris = load_iris()
X = iris.data

# Apply PCA to reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plotting
fig, ax = plt.subplots()
colors = plt.cm.get_cmap('tab10', k)

for cluster in range(k):
    cluster_points = X_pca[y_kmeans == cluster]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               label=f'Cluster {cluster}', s=50, alpha=0.7, color=colors(cluster))

ax.set_title("Clusters (2D PCA Projection)")
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend()

# Show plot
st.pyplot(fig)