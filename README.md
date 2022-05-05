# Text-Clustering
Python and ML project to cluster the descriptions of Siena's courses.

# Project Overview
The goal of this project is to cluster the descriptions of Siena's courses. The first step will be vectorizing the text appropriately. 

Four different groupings will be created:
- A clustering with three groups (corresponding to Siena's three schools)
- A clustering with 33 groups (corresponding to departments)
- A clustering with 57 groups (corresponding to course prefixes)
- A clustering with the optimal number of groups, based on silhouette score

The first three of these have a fixed number of clusters, so I decided to use k-means, agglomerative clustering, and LDA to compare which is best by adjusting parameters to maximize the ARI when compared to the ground truth.

For the fourth case, I used the best parameter settings from above and tried out several different numbers of groups, ranging from 2 up to over 80 groups. I also tried DBSCAN with a few different parameter settings in order to see how many groups it suggested as well. My optimal grouping is the one with the highest silhouette score.

# Data files:
- descriptions.txt- The text of the course descriptions.
- school_codes.txt- The school each ID is associated with.
- dept_codes.txt- The departments each ID is associated with.
- prefix_codes.txt- The course prefix each ID is associated with.
- SCRTXT.xlsx- Process of generating the data files. Also, the "SchoolMap" tab in the Excel workbook shows the mapping from course prefixes to school/department/prefix codes.

### Notes on Data Files
- **Reminder:** The models are built only from the descriptions. The other three files are there so that I could compute an ARI score.
- All of the files are tab-delimited text; each file has only two columns - the ID and the content described.
- All of the IDs and labels have been randomized - there is no information to be mined from them.
- Any label of 0 indicates that the course does not fit cleanly into one of the schools/departments (e.g. military studies courses for ROTC students).

## Setup
To setup your computer, follow these steps:

1) Install Spyder: https://docs.spyder-ide.org/current/installation.html
2) If you need to install any of the packages I used, follow the instructions in this video: https://www.youtube.com/watch?v=i7Njb3xO4Fw
3) Clone this repo using the command:
```
git clone "https://github.com/lwoluke/Text-Clustering.git"
```

