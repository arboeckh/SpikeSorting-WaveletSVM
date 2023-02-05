"""Script to asses effects of false positive rejection"""
import ss 


#hyper params
hpFreq = 50                     #high-pass frequency of the high pass filter
thK = 3                         #threshold K value for spike detector
clipSize = 64                   #sample length of each extracted clip
clipDistDef = [20,100]          #defines range of merged, connected, and isolated clips
training_data_ratio = 0.8       #ratio of data to be used for trainig vs. cross-validation
denoisingWavelet = 'sym4'       #mother wavelet to be used for denoising
denoisingLevel = 1              #decomposition level used for wavelet denoising
denoising_thresh_coeff = 0.5     #coefficient to scale the threshold in wavelet denoising
dwtWavelet = 'sym4'             #mother wavelet to be used for the discrete wavelet transform 
svmKernel='rbf'                 #SVM kernel for SVM classifier
evalTol = 10                    #allowable error between label index and classified index

#get training data
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
    'fp_rejection': False
}
#create spike sorter
spike_sorter = ss.SpikeSorter(**spike_sorter_params)
spike_sorter.train()
spike_sorter.params['thK'] = 1
print("No false positive rejection")
print(f"k: {spike_sorter.params['thK']}, results: {spike_sorter.cross_validate(plot=False)}")

#now train and cross-validate with false positive rejection
spike_sorter.params['fp_rejection'] = True
spike_sorter.train()
print("False positive Rejection")
thks = [1,2,3,4,5,6]
for k in thks:
    spike_sorter.params['thK'] = k
    print(f"k: {k}, results: {spike_sorter.cross_validate(plot=False)}")




















