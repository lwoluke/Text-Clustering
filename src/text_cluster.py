# -*- coding: utf-8 -*-
"""
Cluster the descriptions of Siena's courses

Reminder: Your models should be built ONLY from the descriptions. 
The other three files are there ONLY so that you can compute an ARI score.

@author: Luke Ostrander
"""

import numpy as np

mydir = "Z:/CSIS 320/Projects/Text-Clustering/data/"
descriptions = np.loadtxt(mydir + "descriptions.txt", dtype="str", delimiter="\t", skiprows=1)