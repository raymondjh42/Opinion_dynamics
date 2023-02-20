import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
import numpy as np
import time
import pickle
import random
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
import argparse
import random
import pandas as pd
import networkx as nx
import model_file as mf
import scipy
import glob
import itertools

datasets = [
    '../DATA/snap/facebook_combined.pkl',
    '../DATA/socfb-Mich67/socfb-Mich67.pkl',
    '../DATA/socfb-Wellesley22/socfb-Wellesley22.pkl'
]

d_edges = [0, 2, 4, 8, 16, 32]

for path in datasets:
    graph_features = []
    for e in d_edges:
        graph_features_e = []
        for _ in range(10):
            # Load sparse_matrix.pickle
            with open(path, 'rb') as handle:
                weights = pickle.load(handle)

            epsilon = 0.1
            delta = 0.5
            n = weights.shape[0]
            s = 0.005
            state_init = 'random'

            internal_opinions_1 = np.random.rand(n)
            external_opinions_1 = np.copy(internal_opinions_1)

            model_1 = mf.Model(weights, internal_opinions_1, external_opinions_1, epsilon, delta, s, include_cascade=True, state_init=state_init)
            model_1.over_lay_graph(e)

            graph_features_e.append(model_1.graph_features())
        averaged_graph_features = np.mean(graph_features_e, axis=0)
        graph_features.append(averaged_graph_features)

    # Save in graph_features
    with open('graph_features/' + path.split('/')[-1].split('.')[0] + '.pkl', 'wb') as handle:
        pickle.dump(graph_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Load graph_features_fb_wel.pickle, graph_features_fb_mich.pickle, graph_features_fb.pickle
with open('graph_features_fb_wel.pickle', 'rb') as handle:
    graph_features_fb_wel = pickle.load(handle)
with open('graph_features_fb_mich.pickle', 'rb') as handle:
    graph_features_fb_mich = pickle.load(handle)
with open('graph_features_fb.pickle', 'rb') as handle:
    graph_features_fb = pickle.load(handle)

# Collect third entry of every array into a list
triangles_fb_wel = [graph_features_fb_wel[i][2] for i in range(len(graph_features_fb_wel))]
triangles_fb_mich = [graph_features_fb_mich[i][2] for i in range(len(graph_features_fb_mich))]
triangles_fb = [graph_features_fb[i][2] for i in range(len(graph_features_fb))]

# Collect fourth entry of every array into a list
transitivity_fb_wel = [graph_features_fb_wel[i][3] for i in range(len(graph_features_fb_wel))]
transitivity_fb_mich = [graph_features_fb_mich[i][3] for i in range(len(graph_features_fb_mich))]
transitivity_fb = [graph_features_fb[i][3] for i in range(len(graph_features_fb))]

# Collect fifth entry of every array into a list
clustering_fb_wel = [graph_features_fb_wel[i][4] for i in range(len(graph_features_fb_wel))]
clustering_fb_mich = [graph_features_fb_mich[i][4] for i in range(len(graph_features_fb_mich))]
clustering_fb = [graph_features_fb[i][4] for i in range(len(graph_features_fb))]

# Create 3 horizontal subplots to display the triangle count, transitivity and clustering coefficient
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].plot(d_edges, triangles_fb_wel, label='Wellesley')
axs[0].plot(d_edges, triangles_fb_mich, label='Michigan')
axs[0].plot(d_edges, triangles_fb, label='SNAP')
axs[0].set_title('Triangle count vs edges added')
axs[0].set_xlabel('Edges added')
axs[0].set_ylabel('Triangle count')
axs[0].legend()

axs[1].plot(d_edges, transitivity_fb_wel, label='Wellesley')
axs[1].plot(d_edges, transitivity_fb_mich, label='Michigan')
axs[1].plot(d_edges, transitivity_fb, label='SNAP')
axs[1].set_title('Transitivity vs edges added')
axs[1].set_xlabel('Edges added')
axs[1].set_ylabel('Transitivity')
axs[1].legend()

axs[2].plot(d_edges, clustering_fb_wel, label='Wellesley')
axs[2].plot(d_edges, clustering_fb_mich, label='Michigan')
axs[2].plot(d_edges, clustering_fb, label='SNAP')
axs[2].set_title('Clustering coefficient vs edges added')
axs[2].set_xlabel('Edges added')
axs[2].set_ylabel('Clustering coefficient')
axs[2].legend()

fig = plt.gcf()
fig.suptitle("Graph features vs edges added", fontsize=14)
plt.show()