# Text-Clustering
Python and ML project to cluster the descriptions of Siena's courses.

# Project Overview & Deliverables
The goal of this project is to cluster the descriptions of Siena's courses. The first step will be vectorizing the text appropriately. Make sure to set the max_df parameter of your vectorizer to eliminate extremely common words (e.g. "course").

Four different groupings will be created:
- A clustering with three groups (corresponding to Siena's three schools)
- A clustering with 33 groups (corresponding to departments)
- A clustering with 57 groups (corresponding to course prefixes)
- A clustering with the optimal number of groups, based on silhouette score

The first three of these have a fixed number of clusters, which suggests using k-means, agglomerative clustering, or LDA. (You could try tuning the DBSCAN parameters to generate the correct number of groups, but it's likely to be time-consuming, and I am not requiring it.) For each of these, adjust parameters to maximize the ARI when compared to the ground truth.
- (Note: There are a few courses with "0" labels for department and course prefix, and thus 34 & 58 ground truth labels in these settings. These course do NOT actually group together, and ARI will mark them "wrong" no matter how your model treats them - and that's OK.) 

For the fourth case, use the best parameter settings from above and try several different numbers of groups. (Suggestion: start with 2,4,6,10,15,20,30.) You should also try DBSCAN with a few different parameter settings and see how many groups it suggests as well. Your optimal grouping will be the one with the highest silhouette score.

You will submit two files: your (well-commented) code file, and a report detailing your findings. The report should indicate which model & parameter settings produced the best result in each case as well as your own qualitative analysis that discusses things like which schools/departments show up cleanly and which seem to get merged with other departments or dispersed across several groups. (This is something of a "by hand" process - look through the courses that fall into each grouping.) You might also look at the top terms in each cluster to get a sense of what is important. The report should end up around two pages.

The assignment is due Wednesday, April 27; however, if you need additional time, please ask - I can be flexible (up to a few days) on the deadline.
