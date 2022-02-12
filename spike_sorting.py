"""Script to train and validate the final spike sorter with cross-validation and
self-blurring stability"""
import ss 
import numpy as np
import scipy.io as io

#hyper params to be used for the spike sorter
hpFreq = 50                     #high-pass frequency of the high pass filter
thK = 4                         #threshold K value for spike detector
clipSize = 32                   #sample length of each extracted clip
clipDistDef = [20,100]          #defines range of merged, connected, and isolated clips
training_data_ratio = 0.8       #ratio of data to be used for trainig vs. cross-validation
denoisingWavelet = 'sym4'       #mother wavelet to be used for denoising
denoisingLevel = 4              #decomposition level used for wavelet denoising
denoising_thresh_coeff = 0.4    #coefficient to scale the threshold in wavelet denoising
dwtWavelet = 'haar'             #mother wavelet to be used for the discrete wavelet transform 
svmKernel='rbf'                 #SVM kernel for SVM classifier
evalTol = 10                    #allowable error between label index and classified index

#get training data and submission data
d, training_labels = ss.load_training_data()
sd = ss.load_submission_data()

#create dict to load up spike sorter params into spike sorter object
spike_sorter_params = {
    'data': d,
    'sub_data': sd,
    'hpFreq': hpFreq,
    'Fs': 25e3,
    'denoisingWavelet': denoisingWavelet,
    'denoisingLevel': denoisingLevel,
    'thK': thK,
    'labels': training_labels,
    'training_data_ratio': training_data_ratio,
    'clipSize': clipSize,
    'clipDistDef': clipDistDef,
    'dwtWavelet': dwtWavelet,
    'svmKernel': svmKernel,
    'evalTol': evalTol,
    'denoising_thresh_coeff': denoising_thresh_coeff,
    'fp_rejection': True
}

#instantiate spike sorter object
spike_sorter = ss.SpikeSorter(**spike_sorter_params)

#train spike sorter
spike_sorter.train()

#do self bluring stability and cross-validate
stabilities = spike_sorter.self_blur_stability(1, gamma=0.15)
results = spike_sorter.cross_validate(plot=True, plotFalseNegs=True)
print(f"Total accuracy: {results['total_accuracy']}, stability: {np.mean(stabilities)}")
print(results)

#apply spike sorter on submission data to get Index and Class arrays
sub_labels = spike_sorter.eval_new_data(sd, 25e3)

#create 2 arrays, one for labels and one for indeces
Index = []
Class = []
for i in range(len(sub_labels[0])):
    Index.append(sub_labels[0][i][0])
    Class.append(sub_labels[0][i][1])
sub_data = {"Index": Index, "Class": Class}
io.savemat("submission_mat/13807.mat", sub_data)
pass






















