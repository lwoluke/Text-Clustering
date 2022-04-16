# -*- coding: utf-8 -*-
"""
Cluster the descriptions of Siena's courses

Reminder: Your models should be built ONLY from the descriptions. 
The other three files are there ONLY so that you can compute an ARI score.

@author: Luke Ostrander
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import LatentDirichletAllocation


# Load Training Data
mydir = "C:/Users/lwolu/OneDrive/Documents/College/Siena/Spring 2022 Semester/CSIS 320 - Machine Learning/Projects/Text-Clustering/data/"
file = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
descriptions = file[:,1]

# Experiment with different max_df values in CountVectorizer
desc_vec = CountVectorizer(max_df=0.2).fit_transform(descriptions)

'''
Clusters
A clustering with three groups (corresponding to Siena's three schools)
A clustering with 33 groups (corresponding to departments)
A clustering with 57 groups (corresponding to course prefixes)
A clustering with the optimal number of groups, based on silhouette score

First 3 have a fixed number of clusters
Last is to be determined from testing

Create for K-Means, Agglomerative, and LDA to compare which is best for various tests
'''
kmeans_schools = KMeans(n_clusters=3, random_state=42).fit(desc_vec)
kmeans_departments = KMeans(n_clusters=37, random_state=42).fit(desc_vec)
kmeans_courses = KMeans(n_clusters=57, random_state=42).fit(desc_vec)

km_school_labels = kmeans_schools.labels_
km_departments_labels = kmeans_departments.labels_
km_courses_labels = kmeans_courses.labels_

# agglo_schools = AgglomerativeClustering(n_clusters=3).fit(descriptions_vec)
# agglo_departments = AgglomerativeClustering(n_clusters=37).fit(descriptions_vec)
# agglo_courses = AgglomerativeClustering(n_clusters=57).fit(descriptions_vec)

# lda_schools = LinearDiscriminantAnalysis(n_components=3, random_state=42).fit(descriptions_vec)
# lda_departments = LinearDiscriminantAnalysis(n_components=37, random_state=42).fit(descriptions_vec)
# lda_courses = LinearDiscriminantAnalysis(n_components=57, random_state=42).fit(descriptions_vec)

'''
Use best parameter settings from above and try several different numbers of groups

Suggestion: start with 2,4,6,10,15,20,30
The optimal grouping will be the one with the highest silhouette score
'''
kmeans_best = KMeans(n_clusters=2, random_state=42).fit(desc_vec)
km_best_labels = kmeans_best.labels_

# agglo_best = AgglomerativeClustering(n_clusters=2).fit(descriptions_vec)
# lda_best = LinearDiscriminantAnalysis(n_components=2, random_state=42).fit(descriptions_vec)
