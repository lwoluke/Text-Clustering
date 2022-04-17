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

def main():
    
    ### LOAD TRAINING DATA ###
    mydir = "C:/Users/lwolu/OneDrive/Documents/College/Siena/Spring 2022 Semester/CSIS 320 - Machine Learning/Projects/Text-Clustering/data/"
    descriptions = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)
    descriptions_key = descriptions[:,0]
    descriptions_data = descriptions[:,1]
    
    ### TEST DIFFERENT PARAMETER VALUES IN CountVectorizer (Use max_df=0.2 as baseline) ###
    desc_countvec = CountVectorizer(ngram_range=(1,1), min_df=0.05, max_df=0.6).fit_transform(descriptions_data)
    desc_tfidfvec = TfidfVectorizer(ngram_range=(1,1), min_df=0.05, max_df=0.6).fit_transform(descriptions_data)
    
    
    '''
    Clusters
    A clustering with three groups (corresponding to Siena's three schools)
    A clustering with 33 groups (corresponding to departments)
    A clustering with 57 groups (corresponding to course prefixes)
    A clustering with the optimal number of groups, based on silhouette score
    
    First 3 have a fixed number of clusters
    Last is to be determined from testing
    
    Create for K-Means, Agglomerative, and LDA to compare which is best for various tests
    
    Labels used for ARI and Silhouette score calculations
    Answers is used by LDA for calculating ARI and Silhouette scores
    '''
    ######## K-MEANS ########
    kmeans_schools = KMeans(n_clusters=3, random_state=42).fit(desc_tfidfvec)
    kmeans_departments = KMeans(n_clusters=33, random_state=42).fit(desc_tfidfvec)
    kmeans_prefixes = KMeans(n_clusters=57, random_state=42).fit(desc_tfidfvec)
    
    kmeans_schools_labels = kmeans_schools.labels_
    kmeans_departments_labels = kmeans_departments.labels_
    kmeans_prefixes_labels = kmeans_prefixes.labels_
    
    km_schools_silo = silhouette_score(desc_tfidfvec, kmeans_schools_labels, metric='manhattan')
    km_departments_silo = silhouette_score(desc_tfidfvec, kmeans_departments_labels, metric='manhattan')
    km_prefixes_silo = silhouette_score(desc_tfidfvec, kmeans_prefixes_labels, metric='manhattan')
    
    ######## AGGLOMERATIVE ########
    agglo_schools = AgglomerativeClustering(n_clusters=3).fit(desc_tfidfvec.toarray())
    agglo_departments = AgglomerativeClustering(n_clusters=33).fit(desc_tfidfvec.toarray())
    agglo_prefixes = AgglomerativeClustering(n_clusters=57).fit(desc_tfidfvec.toarray())
    
    agglo_schools_labels = agglo_schools.labels_
    agglo_departments_labels = agglo_departments.labels_
    agglo_prefixes_labels = agglo_prefixes.labels_
    
    agglo_schools_silo = silhouette_score(desc_tfidfvec, agglo_schools_labels, metric='manhattan')
    agglo_departments_silo = silhouette_score(desc_tfidfvec, agglo_departments_labels, metric='manhattan')
    agglo_prefixes_silo = silhouette_score(desc_tfidfvec, agglo_prefixes_labels, metric='manhattan')
    
    ######## LDA ########
    lda_schools = LatentDirichletAllocation(n_components=3, random_state=42).fit(desc_tfidfvec)
    lda_departments = LatentDirichletAllocation(n_components=33, random_state=42).fit(desc_tfidfvec)
    lda_prefixes = LatentDirichletAllocation(n_components=57, random_state=42).fit(desc_tfidfvec)
    
    ### SCHOOLS TO ARRAY ###
    lda_schools_mat = sp.csr_matrix.toarray(desc_tfidfvec)
    lda_schools_topics = np.matmul(lda_schools_mat, np.transpose(lda_schools.components_))
    lda_schools_answers = np.argmax(lda_schools_topics, axis=1)
    lda_silhouette = silhouette_score(desc_tfidfvec, lda_schools_answers, metric='manhattan')
    
    ### DEPARTMENTS TO ARRAY ###
    lda_departments_mat = sp.csr_matrix.toarray(desc_tfidfvec)
    lda_departments_topics = np.matmul(lda_departments_mat, np.transpose(lda_departments.components_))
    lda_departments_answers = np.argmax(lda_departments_topics, axis=1)
    lda_silhouette = silhouette_score(desc_tfidfvec, lda_departments_answers, metric='manhattan')
    
    ### PREFIXES TO ARRAY ###
    lda_prefixes_mat = sp.csr_matrix.toarray(desc_tfidfvec)
    lda_prefixes_topics = np.matmul(lda_prefixes_mat, np.transpose(lda_prefixes.components_))
    lda_prefixes_answers = np.argmax(lda_prefixes_topics, axis=1)
    lda_silhouette = silhouette_score(desc_tfidfvec, lda_prefixes_answers, metric='manhattan')
    
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
    
    ####### BEST FOR CountVectorizer #######
    kmeans_best = KMeans(n_clusters=75, random_state=42).fit(desc_countvec)
    km_best_labels = kmeans_best.labels_
    kmeans_silhouette = silhouette_score(desc_countvec, kmeans_best.labels_, metric='manhattan')
    
    agglo_best = AgglomerativeClustering(n_clusters=75).fit(desc_countvec.toarray())
    agglo_best_labels = agglo_best.labels_
    agglo_silhouette = silhouette_score(desc_countvec, agglo_best_labels, metric='manhattan')
    
    lda_best = LatentDirichletAllocation(n_components=75, random_state=42).fit(desc_countvec)
    desc_mat = sp.csr_matrix.toarray(desc_countvec)
    post_topics = np.matmul(desc_mat, np.transpose(lda_best.components_))
    lda_best_answers = np.argmax(post_topics, axis=1)
    lda_silhouette = silhouette_score(desc_countvec, lda_best_answers, metric='manhattan')
    
    ####### BEST FOR TfidfVectorizer #######
    kmeans_best = KMeans(n_clusters=75, random_state=42).fit(desc_tfidfvec)
    km_best_labels = kmeans_best.labels_
    kmeans_silhouette = silhouette_score(desc_tfidfvec, km_best_labels, metric='manhattan')
    
    agglo_best = AgglomerativeClustering(n_clusters=75).fit(desc_tfidfvec.toarray())
    agglo_best_labels = agglo_best.labels_
    agglo_silhouette = silhouette_score(desc_tfidfvec, agglo_best_labels, metric='manhattan')
    
    lda_best = LatentDirichletAllocation(n_components=75, random_state=42).fit(desc_tfidfvec)
    desc_mat = sp.csr_matrix.toarray(desc_tfidfvec)
    post_topics = np.matmul(desc_mat, np.transpose(lda_best.components_))
    lda_best_answers = np.argmax(post_topics, axis=1)
    lda_silhouette = silhouette_score(desc_tfidfvec, lda_best_answers, metric='manhattan')
    
    ###### COMPUTE ARI SCORE - K-MEANS, AGGLO, AND LDA ######
    
    ### SIENA'S 3 SCHOOLS - EX: adjusted_rand_score(schoolcodes_data, kmeans_schools.labels_) ###
    schoolcodes = np.loadtxt(mydir + "school_codes.txt", dtype="str", delimiter="\t", skiprows=1)
    schoolcodes_key = schoolcodes[:,0]
    schoolcodes_data = schoolcodes[:,1]
    
    kmeans_schools_ari = adjusted_rand_score(schoolcodes_data, kmeans_schools_labels)
    agglo_schools_ari = adjusted_rand_score(schoolcodes_data, agglo_schools_labels)
    lda_schools_ari = adjusted_rand_score(schoolcodes_data, lda_schools_answers)
    
    ### SIENA'S 33 DEPARTMENTS - EX: adjusted_rand_score(deptcodes_data, kmeans_departments.labels_) ###
    deptcodes = np.loadtxt(mydir + "dept_codes.txt", dtype="str", delimiter="\t", skiprows=1)
    deptcodes_key = deptcodes[:,0]
    deptcodes_data = deptcodes[:,1]
    
    kmeans_departments_ari = adjusted_rand_score(deptcodes_data, kmeans_departments_labels)
    agglo_departments_ari = adjusted_rand_score(deptcodes_data, agglo_departments_labels)
    lda_departments_ari = adjusted_rand_score(deptcodes_data, lda_departments_answers)
    
    ### SIENA'S 57 PREFIXES - EX: adjusted_rand_score(prefcodes_data, kmeans_prefixes.labels_) ###
    prefcodes = np.loadtxt(mydir + "prefix_codes.txt", dtype="str", delimiter="\t", skiprows=1)
    prefcodes_key = prefcodes[:,0]
    prefcodes_data = prefcodes[:,1]
    
    kmeans_prefixes_ari = adjusted_rand_score(prefcodes_data, kmeans_prefixes_labels)
    agglo_prefixes_ari = adjusted_rand_score(prefcodes_data, agglo_prefixes_labels)
    lda_prefixes_ari = adjusted_rand_score(prefcodes_data, lda_prefixes_answers)
    
    ####### DBSCAN #######
    
    ### Generate vectorizers ###
    db_countvec = CountVectorizer(ngram_range=(1,1), max_df=0.2).fit_transform(descriptions_data)
    db_tfidfvec = TfidfVectorizer(ngram_range=(1,1), max_df=0.2).fit_transform(descriptions_data)
    
    ### Generate clusters ###
    db1 = DBSCAN(eps=50, min_samples=3, metric="manhattan", n_jobs=-1)
    db2 = DBSCAN(eps=50, min_samples=7, metric="manhattan", n_jobs=-1)
    db3 = DBSCAN(eps=75, min_samples=5, metric="manhattan", n_jobs=-1)
    
    ### Fit to CountVectorizer and TfidfVectorizer ###
    db1.fit(db_countvec)
    db2.fit(db_countvec)
    db3.fit(db_countvec)
    
    ### To use, change all instances of db_countvec to db_tfidfvec ###
    # db1.fit(db_tfidfvec)
    # db2.fit(db_tfidfvec)
    # db3.fit(db_tfidfvec)
    
    ### Store labels for Silhouette Score ###
    db1_labels = db1.labels_
    db2_labels = db2.labels_
    db3_labels = db3.labels_
    
    ### Get silhouette scores - Ex: silhouette_score(db_countvec, db1.labels_, metric='manhattan') ###
    db1_silhouette = silhouette_score(db_countvec, db1_labels, metric='manhattan')
    db2_silhouette = silhouette_score(db_countvec, db2_labels, metric='manhattan')
    db3_silhouette = silhouette_score(db_countvec, db3_labels, metric='manhattan')
    
    
if __name__ == '__main__':
    main()
