import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as spio
import scipy.signal as signal
import numpy as np
from scipy.sparse import data
from sklearn.decomposition import PCA
from sklearn import svm
import seaborn as sns

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
th = 3 * np.median(np.abs(filtd))/0.6745

#choose training (first 80%) and test data (last 20%)
cutoff = int(0.8 * len(filtd))
for i in range(len(labels)):
    if labels[i,0] >= cutoff:
        labelIndex = i
        break
training_data = filtd[:cutoff]
training_labels = labels[:labelIndex]
test_data = filtd[cutoff:]
test_labels = labels[labelIndex:]

training_clips_labeled, mean_firing_times, _, _ = \
    ss.extract_labeled_clips(filtd, th/4, training_labels, distDef, 20, 64, 5)
test_clips_labeled, _, _, _= ss.extract_labeled_clips(filtd, th/2, test_labels, distDef, 20, 64, 5)

#training of SVM
training_clips_labeled = ss.wavelet_decomposition(training_clips_labeled)
features = []
for clip in training_clips_labeled:
    features.append(clip.features)

#exctract 5 components
pca = PCA(n_components=5)
pcas = pca.fit_transform(features)
for i,clip in enumerate(training_clips_labeled):
    clip.pcas = pcas[i]

#Train SVM on training clips
t_data, t_labels = ss.generate_training_data_from_clips(training_clips_labeled)
clf = ss.train_SVM(t_data, t_labels)

#Do spike detection on test data time series
detected_clips = ss.spike_detection(test_data, cutoff, th, windowSize=64, minPeakDist=20)
#extract wavelet coefficients
detected_clips = ss.wavelet_decomposition(detected_clips)
features = []
for clip in detected_clips:
    features.append(clip.features)
#dimension reduction    
pcas = pca.transform(features)
for i,clip in enumerate(detected_clips):
    clip.pcas = pcas[i]

classified_clips = ss.classifiy_SVM_merged_detect(detected_clips, test_clips_labeled, clf, mean_firing_times)

#put clips into list of [index,label] for confusion matrix
classified_clips_toCM = []
for clip in classified_clips:
    classified_clips_toCM.append([clip.spikeTime, clip.label])

qmat = ss.confusion_matrix_unlabeled(classified_clips_toCM, test_labels, 10, 5)
results = ss.evaluate_results_labeled(qmat)
print(results)

plt.figure(figsize=(3.2,3.2))
mask = np.zeros_like(qmat)
mask[np.triu_indices_from(qmat)] = True
labels = ['1','2','3','4','5','null']
ax = sns.heatmap(qmat, xticklabels=labels, yticklabels=labels, annot=True, cbar=False, fmt='g', cmap='coolwarm')
ax.tick_params(axis='both', which='major', labelsize=9)
plt.ylabel("Spike Sorting Neurons")
plt.xlabel("Labeled Data Neurons")
sns.color_palette("mako", as_cmap=True)
plt.title("Spike Sorting Confusion Matrix")
plt.tight_layout()
plt.show()







