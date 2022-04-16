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
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# load training data
mydir = "C:/Users/lwolu/OneDrive/Documents/College/Siena/Spring 2022 Semester/CSIS 320 - Machine Learning/Projects/Text-Clustering/data/"
descriptions = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
descriptions_data = descriptions[:,1]
descriptions_vec = CountVectorizer(max_df=.9).fit_transform(descriptions_data)

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
kmeans_schools = KMeans(n_clusters=3, random_state=42).fit(descriptions_vec)
kmeans_departments = KMeans(n_clusters=37, random_state=42).fit(descriptions_vec)
kmeans_courses = KMeans(n_clusters=57, random_state=42).fit(descriptions_vec)

agglo_schools = AgglomerativeClustering(n_clusters=3).fit(descriptions_vec)
agglo_departments = AgglomerativeClustering(n_clusters=37).fit(descriptions_vec)
agglo_courses = AgglomerativeClustering(n_clusters=57).fit(descriptions_vec)

lda_schools = LinearDiscriminantAnalysis(n_components=3, random_state=42).fit(descriptions_vec)
lda_departments = LinearDiscriminantAnalysis(n_components=37, random_state=42).fit(descriptions_vec)
lda_courses = LinearDiscriminantAnalysis(n_components=57, random_state=42).fit(descriptions_vec)

'''
Use best parameter settings from above and try several different numbers of groups

Suggestion: start with 2,4,6,10,15,20,30
The optimal grouping will be the one with the highest silhouette score
'''
kmeans_schools = KMeans(n_clusters=2, random_state=42).fit(descriptions_vec)
agglo_schools = AgglomerativeClustering(n_clusters=2).fit(descriptions_vec)
lda_schools = LinearDiscriminantAnalysis(n_components=2, random_state=42).fit(descriptions_vec)
