#!/usr/bin/env python
# -*- coding: utf-8 -*-

# IMPORT OF LIBRARIES
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from operator import itemgetter


###################################
####       Preprocessing       ####
###################################

outfile = open("sorted_dataset.csv", "w")
with open("dataset_mood_smartphone.csv", "r") as DM:
    Data = DM.readlines()

#make a list containing strings out of the dataset file
DataList = [x.strip() for x in Data]
#make a list out of each string (line) in the list of strings
DataSet = [x.split(",") for x in DataList]

#remove all excessive double quotes from each element in the list of lists
for i in range(0, len(DataSet)):
    DataSet[i][0] = DataSet [i][0][1:-1]
    DataSet[i][1] = DataSet[i][1][1:-1]
    DataSet[i][3] = DataSet[i][3][1:-1]

#order the dataset based on person, then variable, then timepoint
DataSet.sort(key=itemgetter(1,3,2))

#now that the order is correct, remove hours and minutes from timepoints (we only need the info per day)
for i in range(0, len(DataSet)):
    DataSet[i][2] = DataSet [i][2][0:-13]





#write new processed dataset to file
for item in DataSet:
  outfile.write("%s\n" % item)

print(DataSet[1])