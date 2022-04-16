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
from sklearn.decomposition import LatentDirichletAllocation
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import LatentDirichletAllocation


# Load Training Data
mydir = "C:/Users/lwolu/OneDrive/Documents/College/Siena/Spring 2022 Semester/CSIS 320 - Machine Learning/Projects/Text-Clustering/data/"
descriptions = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
descriptions_key = descriptions[:,0]
descriptions_data = descriptions[:,1]

# Experiment with different max_df values in CountVectorizer (Use 0.2 as baseline)
# Euclidean distance usually doesn't work well with count vectors. You might try it with tf-idf, though.
desc_countvec = CountVectorizer(ngram_range=(2,2), max_df=0.2).fit_transform(descriptions_data)
desc_tfidfvec = TfidfVectorizer(ngram_range=(2,2), max_df=0.2).fit_transform(descriptions_data)


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
# K-MEANS
# kmeans_schools = KMeans(n_clusters=3, random_state=42).fit(desc_countvec)
# kmeans_departments = KMeans(n_clusters=33, random_state=42).fit(desc_countvec)
# kmeans_prefixes = KMeans(n_clusters=57, random_state=42).fit(desc_countvec)

# km_school_labels = kmeans_schools.labels_
# km_departments_labels = kmeans_departments.labels_
# km_prefixes_labels = kmeans_prefixes.labels_

# AGGLOMERATIVE 
# agglo_schools = AgglomerativeClustering(n_clusters=3).fit(desc_countvec.toarray())
# agglo_departments = AgglomerativeClustering(n_clusters=33).fit(desc_countvec.toarray())
# agglo_prefixes = AgglomerativeClustering(n_clusters=57).fit(desc_countvec.toarray())

# agglo_schools_labels = agglo_schools.labels_
# agglo_departments_labels = agglo_departments.labels_
# agglo_prefixes_labels = agglo_prefixes.labels_

# LDA
# lda_schools = LatentDirichletAllocation(n_components=3, random_state=42).fit(desc_countvec)
# lda_departments = LatentDirichletAllocation(n_components=33, random_state=42).fit(desc_countvec)
# lda_prefixes = LatentDirichletAllocation(n_components=57, random_state=42).fit(desc_countvec)

# SCHOOLS TO ARRAY
# lda_schools_mat = sp.csr_matrix.toarray(desc_countvec)
# lda_schools_topics = np.matmul(lda_schools_mat, np.transpose(lda_schools.components_))
# lda_schools_answers = np.argmax(lda_schools_topics, axis=1)

# DEPARTMENTS TO ARRAY
# lda_departments_mat = sp.csr_matrix.toarray(desc_countvec)
# lda_departments_topics = np.matmul(lda_departments_mat, np.transpose(lda_departments.components_))
# lda_departments_answers = np.argmax(lda_departments_topics, axis=1)

# COURSES TO ARRAY
# lda_prefixes_mat = sp.csr_matrix.toarray(desc_countvec)
# lda_prefixes_topics = np.matmul(lda_prefixes_mat, np.transpose(lda_courses.components_))
# lda_prefixes_answers = np.argmax(lda_prefixes_topics, axis=1)

'''
Use best parameter settings from above and try several different numbers of groups

Suggestion: start with 2,4,6,10,15,20,30
The optimal grouping will be the one with the highest silhouette score

When calculating silhouette coefficient:
 The value is between [-1, 1]. A score of 1 denotes the best meaning that the 
 data point i is very compact within the cluster to which it belongs and far away
 from the other clusters. A value near 0 represents overlapping clusters with 
 samples very close to the decision boundary of the neighboring clusters. 
 A negative score indicates that the samples might have got assigned to the 
 wrong clusters.
'''

### BEST FOR CountVectorizer ###
# kmeans_best = KMeans(n_clusters=2, random_state=42).fit(desc_countvec)
# km_best_labels = kmeans_best.labels_
# kmeans_silhouette = silhouette_score(desc_countvec, kmeans_best.labels_, metric='manhattan')

# agglo_best = AgglomerativeClustering(n_clusters=2).fit(desc_countvec.toarray())
# agglo_best_labels = agglo_best.labels_
# agglo_silhouette = silhouette_score(desc_countvec, agglo_best_labels, metric='manhattan')

# lda_best = LatentDirichletAllocation(n_components=2, random_state=42).fit(desc_countvec)
# desc_mat = sp.csr_matrix.toarray(desc_countvec)
# post_topics = np.matmul(desc_mat, np.transpose(lda_best.components_))
# lda_best_answers = np.argmax(post_topics, axis=1)
# lda_silhouette = silhouette_score(desc_countvec, lda_best_answers, metric='manhattan')

### BEST FOR TfidfVectorizer ###
# kmeans_best = KMeans(n_clusters=2, random_state=42).fit(desc_tfidfvec)
# km_best_labels = kmeans_best.labels_
# kmeans_silhouette = silhouette_score(desc_tfidfvec, kmeans_best.labels_, metric='manhattan')

# agglo_best = AgglomerativeClustering(n_clusters=2).fit(desc_tfidfvec.toarray())
# agglo_best_labels = agglo_best.labels_
# agglo_silhouette = silhouette_score(desc_tfidfvec, agglo_best_labels, metric='manhattan')

# lda_best = LatentDirichletAllocation(n_components=2, random_state=42).fit(desc_tfidfvec)
# desc_mat = sp.csr_matrix.toarray(desc_tfidfvec)
# post_topics = np.matmul(desc_mat, np.transpose(lda_best.components_))
# lda_best_answers = np.argmax(post_topics, axis=1)
# lda_silhouette = silhouette_score(desc_tfidfvec, lda_best_answers, metric='manhattan')

'''
Best silhouette scores for:
    
    1.
    ngram_range = (1,1)
    max_df = 0.2
    
    NUM CLUSTERS IDEAL
    CountVectorizer)
    kmeans = 2
    Agglomerative = 5
    lda = 2
    
    TfidfVectorizer)
    kmeans = 2
    Agglomerative = 2
    lda = 50
    
    2.
    ngram_range = (2,2)
    max_df = 0.2
    
    NUM CLUSTERS IDEAL
    CountVectorizer)
    kmeans =
    Agglomerative = 
    lda = 
    
    TfidfVectorizer)
    kmeans = 
    Agglomerative = 
    lda = 
'''

### COMPUTE ARI SCORE - K-MEANS, AGGLO, AND LDA ###

# SIENA'S 3 SCHOOLS - EX: adjusted_rand_score(schoolcodes_data, kmeans_schools.labels_)
schoolcodes = np.loadtxt(mydir + "school_codes.txt", dtype="str", delimiter="\t", skiprows=1)
schoolcodes_key = schoolcodes[:,0]
schoolcodes_data = schoolcodes[:,1]

# SIENA'S 33 DEPARTMENTS - EX: adjusted_rand_score(deptcodes_data, kmeans_departments.labels_)
deptcodes = np.loadtxt(mydir + "dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
deptcodes_key = deptcodes[:,0]
deptcodes_data = deptcodes[:,1]

# SIENA'S 57 PREFIXES - EX: adjusted_rand_score(prefcodes_data, kmeans_prefixes.labels_)
prefcodes = np.loadtxt(mydir + "prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)
prefcodes_key = prefcodes[:,0]
prefcodes_data = prefcodes[:,1]
