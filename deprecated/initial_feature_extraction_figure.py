import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.io as spio
import scipy.signal as signal
import numpy as np
from sklearn.decomposition import PCA

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
labeled_clips = ss.extract_labeled_clips(filtd, th, labels, distDef, 20, 64)

#perform wavelet decomposition on all labeled clips
labeled_clips = ss.wavelet_decomposition(labeled_clips)
features = []
for clip in labeled_clips:
    features.append(clip.features)

#pca with 3 components
pca = PCA(n_components=3)
pcas = pca.fit_transform(features)
for i,clip in enumerate(labeled_clips):
    clip.pcas = pcas[i]


plt.subplot(3,1,1)
plt.title("Labeled Spikes in 2 Wavelet PCs")
plt.ylabel("PC2")

plt.subplot(3,1,2)
# plt.title("Clustering of Connected Spikes in 2PCs")
plt.ylabel("PC2")

plt.subplot(3,1,3)
# plt.title("Clustering of Merged Spikes in 2PCs")
plt.xlabel("PC1")
plt.ylabel("PC2")

for clip in labeled_clips:
    if clip.label == 1:
        color = 'blue'
    elif clip.label == 2:
        color = 'green'
    elif clip.label == 3:
        color = 'red'
    elif clip.label == 4:
        color = 'black'
    elif clip.label == 5:
        color = 'orange'
    if clip.seperationType == "isolated":
        plt.subplot(3,1,1)
        plt.scatter(clip.pcas[0], clip.pcas[1], s=20, color=color)
    if clip.seperationType == "connected":
        plt.subplot(3,1,2)
        plt.scatter(clip.pcas[0], clip.pcas[1], s=20, color=color)
    if clip.seperationType == "merged":
        plt.subplot(3,1,3)
        plt.scatter(clip.pcas[0], clip.pcas[1], s=20, color=color)

plt.subplot(3,1,1)
plt.tight_layout() 
plt.subplot(3,1,2)
plt.tight_layout()
plt.subplot(3,1,3)
plt.tight_layout()
plt.show()







pass