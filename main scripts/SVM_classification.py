"""Script to cross-validate the SVM classifier """
import scipy.io as spio
import numpy as np
from sklearn.decomposition import PCA
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

#perform wavelet decomposition on all labeled clips
labeled_clips = ss.wavelet_decomposition(labeled_clips)
features = []
for clip in labeled_clips:
    features.append(clip.features)

#perfrom 20 training and cross-validation on shuffled deck of training clips using 
#a range of number of principal components
n_comp = [2,3,4,5,6]
fields = ["category", "accuracy", "Number of PCs"]
results = []
for comp in n_comp:
    #perform PCA 
    pca = PCA(n_components=comp)
    pcas = pca.fit_transform(features)
    for i,clip in enumerate(labeled_clips):
        clip.pcas = pcas[i]
    runs = 20
    accuracies = []
    #do 20 randomised cross-validations on the training data
    for i in range(runs):
        #shuffle the clips, use 80 for training
        shuffled_data = ss.get_shuffled_train_test_data(labeled_clips, 0.8)
        accuracy = ss.eval_SVM(shuffled_data)
        #put into long skinny to be used in tableau software for plotting
        results.append(["isolated", accuracy[0], comp])
        results.append(["connected", accuracy[1], comp])
        results.append(["merged", accuracy[2], comp])
        results.append(["total", accuracy[3], comp])

#save data to CSV file
with open('svm_accuracies/svm_accuracies_forPCs.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(results)

