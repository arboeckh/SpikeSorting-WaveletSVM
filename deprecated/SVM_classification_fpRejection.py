import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as spio
import scipy.signal as signal
import numpy as np
from scipy.sparse import data
from sklearn.decomposition import PCA
from sklearn import svm

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
labeled_clips, mean_firing_times,_,_ = ss.extract_labeled_clips(filtd, th, labels, \
    distDef, 64, 5)
fpClips = ss.extract_fp_clips(filtd, len(filtd), labeled_clips, th, 64, distDef[0])

labeled_clips.extend(fpClips)

#perform wavelet decomposition on all labeled clips
labeled_clips = ss.wavelet_decomposition(labeled_clips)
features = []
for clip in labeled_clips:
    features.append(clip.features)

results = []
pca = PCA(n_components=5)
pcas = pca.fit_transform(features)
for i,clip in enumerate(labeled_clips):
    clip.pcas = pcas[i]
runs = 20
accuracies = []
for i in range(runs):
    #shuffle the clips, use 80 for training
    shuffled_data = ss.get_shuffled_train_test_data(labeled_clips, 0.8)
    accuracy = ss.eval_SVM_withoutClipType(shuffled_data)
    #put into long skinny to be used in tableau
    results.append(["accuracy", accuracy])
      
with open('svm_accuracies_fpRejection.csv', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
    
    write.writerows(results)

