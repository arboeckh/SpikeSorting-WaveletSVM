"""Script to asses success rate of labeled data clip extraction"""

import scipy.io as spio
import numpy as np

import csv

import ss 

#get data
mat = spio.loadmat('data/training.mat', squeeze_me=True) 
d = mat['d']
Index = mat['Index'] 
Class = mat['Class']
labels = []
for i in range(len(Index)):
    labels.append([Index[i],Class[i]])
labels = np.array(sorted(labels, key=lambda x: x[0]))

#filter and generate training clips from labeled data
distDef = [20,100]
filtd = ss.wdn(d, 50, 25e3)
th =  np.median(np.abs(filtd))/0.6745
labeled_clips, mean_firing_times, false, (isolated,connected,merged) = \
    ss.extract_labeled_clips(filtd, th, labels, distDef, 64, 5)

#display infomation of extracted clips
print(f"Removed {false} of {len(labeled_clips)} clips -> clip extraction rate: {(len(labeled_clips)-false)/len(labeled_clips)}")
print(f"Isolated: {isolated/len(labeled_clips)*100}%, \
    Connected: {connected/len(labeled_clips)*100}%, \
        Merged: {merged/len(labeled_clips)*100}%")



