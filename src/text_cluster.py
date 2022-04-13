# -*- coding: utf-8 -*-
"""
Cluster the descriptions of Siena's courses

Reminder: Your models should be built ONLY from the descriptions. 
The other three files are there ONLY so that you can compute an ARI score.

@author: Luke Ostrander
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# load training data
mydir = "Z:/CSIS 320/Projects/Text-Clustering/data/"
descriptions = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)

'''
KMeans clusters
A clustering with three groups (corresponding to Siena's three schools)
A clustering with 33 groups (corresponding to departments)
A clustering with 57 groups (corresponding to course prefixes)
A clustering with the optimal number of groups, based on silhouette score

First 3 have a fixed number of clusters
'''
kmeans_schools = KMeans(n_clusters=3, random_state=0).fit(descriptions)
kmeans_departments = KMeans(n_clusters=37, random_state=0).fit(descriptions)
kmeans_schools = KMeans(n_clusters=57, random_state=0).fit(descriptions)
