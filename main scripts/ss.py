"""Spike Sorting Class and auxiliary functions for a wavelet decomposition-based SVM 
Classifier"""
import scipy.io as spio
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.utils import shuffle
import seaborn as sns
import copy

#------------------------------------------------------------------------------------#
#-----------------Final Spike Sorter Class and Loading Functions---------------------#
#------------------------------------------------------------------------------------#
def load_training_data():
    #get data
    mat = spio.loadmat('data/training.mat', squeeze_me=True) 
    d = mat['d']
    Index = mat['Index'] 
    Class = mat['Class']
    labels = []
    for i in range(len(Index)):
        labels.append([Index[i],Class[i]])
    return d, np.array(sorted(labels, key=lambda x: x[0]))

def load_submission_data():
    #get data
    mat = spio.loadmat('data/submission.mat', squeeze_me=True) 
    return mat['d']

class Clip:
    """Class to store data and metadata of clips"""
    def __init__(self, peakIndex, data, indexRange, spikeTime=None, \
        label=None, seperationType=None, features=None, pcas=None):
        self.peakIndex = peakIndex
        self.data = data
        self.indexRange = indexRange
        #if spike time unknown, will be assigned by classifier later
        self.spikeTime = spikeTime  
        self.label = label  #can be set if label is known
        self.seperationType = seperationType
        self.features = features
        self.pcas = pcas

class SpikeSorter:

    def __init__(self, data, sub_data, hpFreq, Fs, denoisingWavelet, denoisingLevel, \
        denoising_thresh_coeff, thK, labels, training_data_ratio, \
    clipSize, clipDistDef, dwtWavelet, svmKernel, evalTol, fp_rejection):
        """Spike Sorter Class based on wavelet decomposition and SVM classification
        INPUTS:
            data: training data
            sub_data: submission data
            hp_freq: high-pass frequency of the initial filtering
            Fs: sampling frequency of the data
            denoisingWavelet: mother wavelet to be used for wavelet denoising
            denoisingLevel: level of decomposition to be used for waveelt denoising
            denoising_thresh_coeff: coefficient to scale the wavelet denoising
            thK: spike detector threshold coefficient
            labels: list of training labels wiht each entry [index, label]
            training_data_ratio: ratio of training data to be used for training, 
                the rest being used for cross-validation
            clipSize: length of the data clip extracted around each peak. Should be
                a power of 2 for efficiency
            clipDistDef: list [a,b] where a is minimum peak to peak distance that is
                detected by the peak detector, and b is the minimum distance between
                spikes for them to be considered isolated
            dwtWavelet: mother wavelet to be used for the discrete wavelet transform.
                Must be suitable for discrete wavelet transfrom
            svmKernel: kernel to be used by the SVM classifier. Can be 'linear', 
                'poly', 'rbf', or 'sigmoid' 
            evalTol: tolerance between detected clips and labeled clips for them to 
                be considered the same
            fp_rejection: bool stating if false positive rejection is to be trained 
            into the SVM classifier"""
        #save all params
        self.params = locals()

        #filter the training data
        self.filtd, self.th = self.filter(data, Fs)
 
        #extract all labeled clips for training
        self.training_data_params = {
        'data': self.filtd,
        'labels': labels,
        'training_ratio': training_data_ratio,
        'clipSize': clipSize,
        'noCat': 5,
        'clipDistDef': clipDistDef,
        'clipDetectThreshold': self.th/4
        }
        self.training_clips, self.mean_firing_times, \
        self.test_data, self.test_labels, self.cutoffIndex = \
            self.gen_training_test_data(**self.training_data_params)

        #extract false positive clips
        self.fpClips = self.extract_fp_clips(self.filtd, self.cutoffIndex, \
            self.training_clips, self.th/4, self.params['clipSize'], self.params['clipDistDef'][0])
        
        #find mean waveforms for all categories
        self.mean_waveforms = self.extract_mean_waveforms(\
            self.training_clips, self.fpClips, noRealCat=self.training_data_params['noCat'])

    def filter(self, data, fs):
        """Function to filter as signal using wavelet denoising and finds the
        peak detection threshold based on the variance of the filtered signal
        INPUTS:
            data: data to be filtered
            fs: sampling frequency of the data
        OUTPUTS:
            filtd: filtered data
            th: threshold for peak detection"""
        #filter the signal using wavelet denoising
        filtd = self.wdn(data, self.params['hpFreq'], fs, \
            wavelet=self.params['denoisingWavelet'], \
            thresh_scaling=self.params['denoising_thresh_coeff'],\
            level = self.params['denoisingLevel'])
        #calculate threshold based on estimate variance of signal
        th = self.calc_threshold(filtd)
        return filtd, th

    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)

    def wdn(self, data, fclow, fs, wavelet='sym4', level=1, thresh_scaling=0.75):
        """Wavelet denoising of a signal with subsequenct high-pass filtering
        INPUTS:
            data: data to be filtered
            fclow: high-pass filter cutoff frequency
            fs: sampling frequency of the data
            wavelet: mother wavelet to be used
            level: wavelet decomposition level
            thresh_scaling: scaling coefficient to scale the threshold value applied
                to the wavelet coefficients
        OUTPUTS:
            filtered signal
        Code modified from: 
        https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform"""
        coeffs = pywt.wavedec(data, wavelet, mode='per')
        sigma = (1/0.6745) * self.madev(coeffs[-level])
        uthresh = thresh_scaling * sigma * np.sqrt(2 * np.log(len(data)))
        coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])
        filt_wvt = pywt.waverec(coeffs, wavelet, mode='per')
        return hpf(filt_wvt, fclow, fs, order=2)


    def calc_threshold(self, data):
        """Calculate the recommended threshold for spike detection based on 
        the variance of a set of data
        INPUTS:
            data: data which is to undergo peak detection
        OUTPUTS:
            recommended threshold"""
        return self.params['thK'] * np.median(np.abs(data))/0.6745

    def gen_training_test_data(self, data, labels, training_ratio, \
        clipSize, noCat, clipDistDef, clipDetectThreshold):
        """Generates training and test data from the training dataset
        INUTS:
            data: training data from which clips are to be extracted
            labels: labels of spikes in the training data, list of form [index, label]
            training_ratio: ratio of data to be used for training 
            clipSize: length of the clip to be extracted for each label
            noCat: number of classification categories
            clipDistDef: list [a,b] where a is minimum peak to peak distance that is
                detected by the peak detector, and b is the minimum distance between
                spikes for them to be considered isolated
            clipDetectThreshold: threshold used for peak detection for the purposes
                of finding labeled peaks"""
        #ensure training ratio is valid
        assert training_ratio>0 and training_ratio<=1

        #based on training_ratio, seperate training data and labels into 
        # test and training sets
        cutoff = int(training_ratio * len(data))
        for i in range(len(labels)):
            if labels[i,0] >= cutoff:
                labelIndex = i
                break
        training_data = data[:cutoff]
        training_labels = labels[:labelIndex]
        test_data = data[cutoff:]
        test_labels = labels[labelIndex:]

        #extract labeled clips with their mean firing times per label type
        training_clips, mean_firing_times, _, _ = \
            self.extract_labeled_clips(training_data, clipDetectThreshold, training_labels, \
                clipDistDef, clipSize, noCat)
        return training_clips, mean_firing_times, test_data, test_labels, cutoff

    def extract_fp_clips(self, data, cutoffIndex, labeled_clips, th, windowSize, minPeakDist):
        """identifies false positive clips extracted by spike detection by comparison 
        tot he extracted labeled clips"""
        #detect peaks using spike detector
        clips = self.spike_detection(data[0:cutoffIndex], 0, th, minPeakDist)
        #assume that the spiek time is about 10 samples before peak (by experience)
        for clip in clips:
            clip.spikeTime = clip.peakIndex - 10 
        #go through each clip, if not within 10 of any labeled data, it is false positive
        fp_clips = []
        for clip in clips:
            goodClip = False
            #check to see if spike time is within 10 samples of any of the labeled clips
            for lclip in labeled_clips:
                if np.abs(lclip.spikeTime - clip.spikeTime)<10:
                    goodClip = True
                    break
            #if no matching clip is found, add it to the false positives list
            if not goodClip:
                clip.label = 0
                #before appending, check it is proper length (bug fix)
                if len(clip.data) == self.params['clipSize']: #fix bug where some clips wihtout data slipped through
                    fp_clips.append(clip)
        return fp_clips

    def extract_labeled_clips(self, data, th, labels, distDef, clipSize, nCat):
        """Extract a list of Clip objects for each labeled spike in the data
        INPUTS:
            data: data from which clips are to be extracted
            th: threshold to be used for peak detection
            labels: list of labeled spikes in data of format [index, label] 
            distDef: list [a,b] where a is minimum peak to peak distance that is
                detected by the peak detector, and b is the minimum distance between
                spikes for them to be considered isolated
            clipSize: length of the clip to be extracted for each label
            nCat: number of classification categories
        OUTPUTS: correct_clips, mean_firing_times, false_clips, (isolated,connected,merged)
            correct_clips: list of correctly identified Clips
            mean_firing_times: list of mean firing times for each label category
            false_clips: number of false clips which were rejected
            clipTypeNumbers: form (isolated, connected, merged) is a tuple containg 
                the number of extracted clips per clip type"""
        
        #initialise list to append data
        clips = []
        peak = []
        firing_to_peak = []
        for i in range(nCat):
            firing_to_peak.append([])

        #go through all labels and append a clip for each
        for i in range(len(labels)): 

            #-----------Assign clip type: isolated, connected, merged----------------#
            
            if i==0:
                #for first label, check only distance to second label 
                dist = labels[i+1,0] - labels[i,0]
            elif i==len(labels)-1:
                #for last label, check only distance to previous
                dist = labels[i,0] - labels[i-1,0]
            else:
                #check distance to previous label
                distBack = labels[i,0] - labels[i-1,0]
                #check distance to next label
                distFront = labels[i+1,0] - labels[i,0]
                #smallest distance between the two is chosen as clip type
                if distBack <= distFront:
                    dist = distBack
                else:
                    dist = distFront
            
            #now assign the label its clip type
            if dist < distDef[0]:
                clipType = 2
            elif dist < distDef[1]:
                clipType = 1
            else:
                clipType = 0
            
            #--------------------Perform peak detection------------------------------#

            #find all peaks
            peaks, _ = signal.find_peaks(data[labels[i,0]:labels[i,0]+distDef[0]], \
                height=th, distance=4)
            
            #check if no peaks were detected
            if len(peaks) < 1:
                #if not peak was detected, create a clip with spike time, data will
                #be assigned later based on the mean firing times of its label type
                clips.append(Clip(None, None, None, \
                     spikeTime=labels[i,0], label=labels[i,1], seperationType=2))
                continue
            #if peaks were detected, take the closest peak
            else:
                peak.append(peaks[0]+labels[i,0])     

            #define clip window based on clip size, centred around the peak
            start = peak[-1] - int(clipSize/2)
            end = peak[-1] + int(clipSize/2)

            #create a clip and append to the clip list
            clips.append(Clip(peak[-1], data[start:end], [start,end], \
                spikeTime=labels[i,0], label=labels[i,1], seperationType=clipType))
            #append firing_to_peak time to its respective label so the average 
            # can later be calculated
            time_to_peak = clips[-1].peakIndex-clips[-1].spikeTime
            firing_to_peak[clips[-1].label-1].append(time_to_peak)
        
        #compute mean firing times
        mean_firing_times = []
        for times in firing_to_peak:
            row = np.array(times)
            mean_firing_times.append(int(np.mean(row)))

        #Go through all clips with no found peaks and find the peak based on
        #the label's average firing time
        for clip in clips:
            #check if clip has unsassigned peak time
            if clip.data is None:
                clip.peakIndex = clip.spikeTime + mean_firing_times[clip.label-1]
                start = clip.peakIndex - int(clipSize/2)
                end = clip.peakIndex + int(clipSize/2)
                clip.data = data[start:end]
                clip.indexRange = [start,end]
        
        #verify that all clips extracted indeed correspond to a real label
        false_clips = 0
        correct_clips = []
        for clip in clips:
            pred_firing_time = clip.peakIndex - mean_firing_times[clip.label-1]
            if np.abs(pred_firing_time - clip.spikeTime) > 10:
                false_clips += 1
            else:
                correct_clips.append(clip)

        #calculate the number of each clip type, for tracking purposes
        merged = 0
        connected = 0
        isolated = 0
        for clip in correct_clips:
            if clip.seperationType == 2:
                merged += 1
            elif clip.seperationType == 1:
                connected += 1
            elif clip.seperationType == 0:
                isolated += 1

        return correct_clips, mean_firing_times, false_clips, (isolated,connected,merged)

    def extract_mean_waveforms(self, labeled_clips, fp_clips, noRealCat):
        """Calculate the mean waveforms of label categories and false positives
        INPUTS:
            labeled_clips: all training labeled training clips
            fp_clips: detected false positive clips
            noRealCat: number of label categories
        OUTPUTS:
            mean_waveforms: a 2-D contianing the mean waveforms of all clip labels
                each of the defined clip size in the SpikeSorter class"""
        #Create arrays for calculations
        mean_waveforms = np.zeros((noRealCat+1, self.params['clipSize']))
        clipCounts = np.zeros(noRealCat+1)
        #Compute sum of waveforms of all real neuron labels
        for clip in labeled_clips:
            mean_waveforms[clip.label] += clip.data
            clipCounts[clip.label] += 1 
        #compute sum of waveform of false positives
        for fp_clip in fp_clips:
            mean_waveforms[0] += fp_clip.data
            clipCounts[0] += 1
        #calculate mean for all labels
        for i,waveform in enumerate(mean_waveforms):
            waveform /= clipCounts[i]
        return mean_waveforms

        

    def spike_detection(self, data, startIndex, threshold, minPeakDist):
        """Performs spike detection on the input by thresholding. Steps:
        1. identify spikes by spike detection above a certain threshold and with minimum
        peak distance
        2. extract a window of 64 samples centred around the peak
        3. return a list of Clip objects each containing a clip of a spike
        INPUTS:
            data: data containing the APs
            startIndex: index from which spik detection will be applied to the data
            threshold: thrshold above which an AP is detected
            windowSize: window size around the peak to be taken as a clip
            minPeakdist: minimum distance between peaks in a clip to be 
            considered different
        OUTPUTS:
            clips: a list of clip objects, one for each spike clip
        """
        #do peak detection based on input params
        peaks,_ = signal.find_peaks(data, height=threshold, distance=minPeakDist)
        clips = []
        distFromPeak = int(self.params['clipSize']/2)
        #Go through each peak and create a clip centres around it
        for peak in peaks:
            start = peak - distFromPeak
            end = peak + distFromPeak 
            peak += startIndex
            clips.append(Clip(peak, data[start:end], [start+startIndex,end+startIndex]))
            #check that the data in each clip is the right length, if not delete it (bug fix)
            if len(clips[-1].data)!=self.params['clipSize']:
                clips.pop(-1)
        return clips

    def train(self):
        """Wrapper function to train_spike_sorter"""
        #if using false positive rejection, append the false positive clips
        #to the training data
        training_clips = copy.copy(self.training_clips)
        if self.params['fp_rejection'] == True:
            training_clips.extend(self.fpClips)
        #train the spike sorter and assign the pca and classifier objects to self
        self.pca, self.clf, _ = \
            self.train_spike_sorter(training_clips, self.params['dwtWavelet'], \
                self.params['svmKernel'])

    def train_spike_sorter(self, training_clips, dwtWavelet, svmKernel):
        """PCA and SVM training from training data
        INPUTS:
            training_clips: list of clips to be used for training
            dwtWavelet: mother wavelet of the discrete wavelet transform 
            svmKernel: kernel to be used by the SVM classifier. Can be 'linear',
                'poly', 'rbf', or 'sigmoid'
        OUTPUTS:
            pca: pca object to be used for PC extraction on futur data
            clf: SVM classifier object
            training_clips: list input trianing clips with their features appended"""
        #extract the wavelet coefficients of labeled clips by discrete wavelet transform
        training_clips = self.wavelet_decomposition(training_clips, wavelet=dwtWavelet)
        features = []
        for clip in training_clips:
            features.append(clip.features)

        #perform PCA on dwt features and extract 5 components 
        pca = PCA(n_components=5)
        pcas = pca.fit_transform(features)
        for i,clip in enumerate(training_clips):
            clip.pcas = pcas[i]

        #Train SVM on training clips
        #helper function to extract training data in format accepted by SVM
        t_data, t_labels = self.generate_training_data_from_clips(training_clips)
        clf = self.train_SVM(t_data, t_labels, svmKernel)

        return pca, clf, training_clips

    def wavelet_decomposition(self, clips, wavelet='sym4'):
        """Perform dicrete wavelet transform on the input data
        INPUTS:
            clips: clips on which to apply the dwt
            wavelet: mother wavelet to use for the dwt
        OUTPUTS:
            clips: input clips with their wavelet decomposition added to them"""
        #compute maximum useful wavelet decomposition level
        decomp_level = pywt.dwt_max_level(len(clips[0].data), wavelet)
        #for every clip perform dwt and add the resulting coefficients to the clip object
        for clip in clips:
            coeffs = pywt.wavedec(clip.data, wavelet, level=decomp_level)
            #flaten the coefficients into 1D
            coeffs_flat = []
            for row in coeffs:
                coeffs_flat.extend(row)
            coeffs = np.array(coeffs_flat)
            clip.features = coeffs
        return clips    
            
    def generate_training_data_from_clips(self,clips):
        """Helper function to output data and labels from clips into a format accepted
        by the SVM classifier:
        INPUTS: 
            clips: clips from which to extract the SVM training data
        OUPUTS:
            data: list data of each clip
            labels: corresponding label to each entry in data"""
        #create a data and labels list and append the respective pcs and labels
        data = []
        labels = []
        for clip in clips:
            data.append(clip.pcas)
            labels.append(clip.label)   
        return data,labels

    def train_SVM(self, training_data, training_labels, svmKernel):
        """Create and train an SVM classifier
        INPUTS: 
            training_data: list of PCs of training data
            training_labels: list of labels to corresponding training_data entries
            svmKernel: SVM kernel to be used for the SVM classifier
        OUTPUTS:
            clf: trained SVM classifier object"""
        clf = svm.SVC(kernel=svmKernel)
        clf.fit(training_data, training_labels)
        return clf


    def cross_validate(self, plot=True, plotFalseNegs=False):
        """Perform cross-validation using the training data
        INPUTS:
            plot: bool, if True, plot the confusion matrix
            plotFalseNegs: bool, if True, plot false negative waveforms
        OUTPUTS:
            results: dict with keys:
                        false_pos: number of false positives
                        false_pos_removed: number of false positives removed by the classifier
                        false_neg: number of false negatives
                        correct_classified: number of clips correctly classified
                        miss_classified: number of true clips that were misclassified
                        total_accuracy: total accuracy of the spike sorter"""

        #generate the parameter dict for the spike sort function. Parameters are taken 
        #from self and were generated during class initialisation
        spike_sorting_params = {
            'pca': self.pca, 
            'clf': self.clf,
            'test_data': self.test_data, 
            'mean_firing_times': self.mean_firing_times,
            'test_data_startIndex': self.cutoffIndex, 
            'th': self.calc_threshold(self.test_data), 
            'clipSize': self.params['clipSize'], 
            'minPeakDist': self.params['clipDistDef'][0],
            'dwtWavelet': self.params['dwtWavelet']
        }
        #apply spike sorting
        classified_labels, fp_removed = self.spike_sort(**spike_sorting_params)

        #create confusonmatrix and evaluate it
        qmat, false_negatives = self.confusion_matrix(classified_labels, self.test_labels, \
            self.params['evalTol'], 5)
        results = self.evaluate_results_labeled(qmat)
        results['fp_removed'] = fp_removed

        #if plotting false negative waveforms, plot them now
        if plotFalseNegs:
            self.plot_false_neg(false_negatives)

        #if plotting the confusion matrix
        if plot:
            self.plot_conf_matrix(qmat)

        return results

    def plot_false_neg(self, false_negatives):
        """Helper function to plot false negatives from cross-validation
        INPUTS:
            false_negatives: list of false negative clips
        OUTPUTS:"""
        clipSize = self.params['clipSize']

        #plot each false negative with all mean waveforms for comparison
        for fn in false_negatives:
            label = fn[1]
            time_to_peak_true = self.mean_firing_times[label-1]
            start = fn[0]
            end = fn[0] + clipSize
            plt.figure(figsize=(4,3))
            plt.title(f"True label: {label}")
            plt.plot(self.filtd[start:end])
            
            for i,mean_wf in enumerate(self.mean_waveforms):
                if i==0:
                    continue
                if i == label:
                    alpha = 1
                else:
                    alpha = 0.25
                time_to_peak = self.mean_firing_times[i-1]
                plt.plot(mean_wf[(int(clipSize/2)-time_to_peak):], alpha=alpha)
            plt.legend(["data", "cat1","cat2","cat3","cat4","cat5"])
            plt.axvline(time_to_peak_true, color='black')
            plt.show()

    def confusion_matrix(self, classified_1, classified_2, sample_tol, no_categories):
        """creates a confusion matrix between 2 sets of classification
        INPUTS:
            classified_1,2 = list of classified data with each entry of form [index, label]
            sample_tol: sample tolerance for 2 data entries of classified_1 and
                        classified_2 to be considered the same
            no_categories: number of categories used for classification
        OUPUTS:
            Qmat: the confusion matrix:
            false_negatives: list of clips that were identified as false negatives"""
        c1 = np.array(classified_1)
        c2 = np.array(classified_2)
        #to track if entries in classified 2 have been associated 
        c2_associated = np.zeros(c2.shape[0])   

        #create confusion matrix
        Qmat = np.zeros([no_categories+1, no_categories+1])

        #Go through c1, associate all c2 within tolerance, and fill in confusion matrix
        for entry in c1:
            c1_index = entry[0]
            c1_category = entry[1]
            associated = np.where((c2[:,0]>=c1_index-sample_tol) \
                & ((c2[:,0]<=c1_index+sample_tol))) 
            associated = associated[0]  #take c1_index samples

            #no associated points found in c2, add to null column for c2
            if len(associated)<1 :
                Qmat[c1_category-1,-1] += 1
            else:
                #if just one associated found, then associate them
                if len(associated) == 1:
                    c2_index = associated[0]
                #check if multiple associated spikes were detected, and choose closest
                elif len(associated)>1:
                    c2_index = associated[0]
                    ds = np.abs(c1_index - c2[c2_index,0])
                    for i in associated[1:]:
                        if np.abs(c1_index - c2[i,0]) < ds:
                            c2_index = i
                            ds = np.abs(c1_index - c2[i,0])
                #Check if associated c2 has alredy been associated, if not associate and 
                #fill the confusion matrix
                if c2_associated[c2_index] == 0:
                    c2_associated[c2_index] = 1
                    c2_category = c2[c2_index,1]
                    Qmat[c1_category-1,c2_category-1] += 1
        
        #go through c2 and find all non-associated entries and add in null row of c1
        false_negatives = []
        for i,asso in enumerate(c2_associated):
            #check if not been associated
            if asso == 0:
                false_negatives.append(c2[i])
                c2_category = c2[i,1]
                #place in confusion matrix
                Qmat[-1,c2_category-1] += 1

        return Qmat, false_negatives

    def plot_conf_matrix(self, qmat):
        """Helper function to plot a confusion matrix:
        INPUTS:
            qmat: confusion matrix to be plotted"""
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

    def evaluate_results_labeled(self, Qmat):
        """function to evaluate classified vs labeled data for the spike sorter
        INPUTS:
            Qmat: confusion matrix using as classification 1 the classified data and 
                classification 2 as the labeled data
        OUTPUTS:
            results: labeled dict containg the evaluation parameters calculated from 
                the confusion matrix
                ->"false_pos": false positives, identified spikes with used classifier but not 
                    found in labeled data
                ->"false_neg": false negatives, identified spikes in labeled data not identified
                    in the classified data
                ->"correct_classified": total number of correctly classified spikes
                ->"miss_classified": true clips that were misclassified
                ->"total_accuracy": total accuracy, the sum of correctly classified spikes (diagonal
                    values) divided by the sum of the all entries in the confusion matrix
                    """

        false_pos = np.sum(Qmat[:,-1])  #sum of null col
        false_neg = np.sum(Qmat[-1,:])  #sum of null col
        trueClips = Qmat[:-1,:-1]
        #correctly classified is the sum of diagonal values
        correct_classified = np.multiply(trueClips, np.eye(len(trueClips))).sum()
        #misclassfied is the sum of all non-null col/row minus the correctly classified
        miss_classified = trueClips.sum() - correct_classified
        #total accuracy is correclty classfied / sum of all entries
        total_accuracy =  correct_classified / np.sum(Qmat)
        return {
            'false_pos': false_pos,
            'false_neg': false_neg,
            'correct_classified': correct_classified,
            'miss_classified': miss_classified,
            'total_accuracy': total_accuracy
        }

    # def plot_conf_matrix(qmat):
    #     """Helper function to plat the confusion matrix
    #     INPUTS:
    #         qmat: confusion matrix to be ploted"""
    #     #plot the confusion matrix using a heatmap 
    #     plt.figure(figsize=(3.2,3.2))
    #     mask = np.zeros_like(qmat)
    #     mask[np.triu_indices_from(qmat)] = True
    #     labels = ['1','2','3','4','5','null']
    #     ax = sns.heatmap(qmat, xticklabels=labels, yticklabels=labels, annot=True, cbar=False, fmt='g', cmap='coolwarm')
    #     ax.tick_params(axis='both', which='major', labelsize=9)
    #     plt.ylabel("Spike Sorting Neurons")
    #     plt.xlabel("Labeled Data Neurons")
    #     sns.color_palette("mako", as_cmap=True)
    #     plt.title("Spike Sorting Confusion Matrix")
    #     plt.tight_layout()
    #     plt.show()


    def stability_analysis(self, Qmat):
        """Performs a stability analysis for each category using a confusion matrix
        INPUTS:
            Qmat: confusion matrix of two different classifications
        OUTPUTS:
            stabilities: array of computed stability for each category"""
        stabilities = np.zeros(Qmat.shape[0]-1)
        #compute stability of each category (excluding null column and row)
        for k in range(Qmat.shape[0]-1):
            stabilities[k] = 2 * Qmat[k,k] / (np.sum(Qmat[k,:]) + np.sum(Qmat[:,k]))
        return stabilities

    def spike_sort(self, pca, clf, test_data, mean_firing_times, \
        test_data_startIndex, th, clipSize, minPeakDist, dwtWavelet):
        """Spike sorting function to be performed on new time series data
        INPUTS:
            pca: PCA object for dimentioanlity reduction
            clf: SVM classifier
            test_data: time series data to be spike sorted
            mean_firing_times: mean number of samples from spike time to peak time per neuron type
            test_data_startIndex: index from which to classify the test data
            th: spike detection threshold value
            clipSize: length of the clips to extract around each spike peak
            minPeakDist: minimum number of samples between adjacent peaks for spike detection
            dwtWavelet: wavelet name to be used for the dwt
        OUTPUTS:
            classified_clip_labels: list of classified spikes of format [index, label]
            rejected_fp: number of false positives rejected by the classifier"""
        #Do spike detection on test data time series
        detected_clips = self.spike_detection(test_data, test_data_startIndex, \
            th, minPeakDist=minPeakDist)
        #extract wavelet coefficients
        detected_clips = self.feature_extraction(detected_clips, dwtWavelet, pca)

        classified_clips, rejected_fp = self.classify_SVM(detected_clips, clf, mean_firing_times)

        #put clips into list of [index,label] for confusion matrix
        classified_clips_labels= []
        for clip in classified_clips:
            classified_clips_labels.append([clip.spikeTime, clip.label])

        return classified_clips_labels, rejected_fp

    def classify_SVM(self, clips, clf: svm.SVC, mean_firing_times):
        """Classify clips using the trained SVM classifier
        INPUTS:
            clips: list of clips to be classified
            clf: SVM classifier object
            mean_firing_times: mean number of indeces from spike to peak or each neuron type
        OUTPUTS:
            output_clips: a list of classified clips"""
        #create a list of the PCs to have proper input format for classifier
        pcs = []
        for clip in clips:
            pcs.append(clip.pcas)
        #perfrom predictions
        predictions = clf.predict(pcs)

        #for each clip, append label. If false positive, do not append to output list
        output_clips = []
        rejected_fn = 0
        for i,clip in enumerate(clips):
            clip.label = predictions[i]
            if clip.label == 0:      #false positive label, so do not append
                rejected_fn += 1
                continue
            clip.spikeTime = clip.peakIndex - mean_firing_times[clip.label-1]
            output_clips.append(clip)
        return output_clips, rejected_fn

    def feature_extraction(self, detected_clips, dwtWavelet, pca):
        """Extract dwt coefficients and perform PCA feature extraction
        INPUTS:
            detected_clips: clips on which to perform feature extraction
            dwtWavelet: mother wavelet used by the dwt
            pca: PCA object to perform feature extraction"""
        #extract wavelet coefficients
        detected_clips = self.wavelet_decomposition(detected_clips, wavelet=dwtWavelet)
        #Make list of wavelet coefficient features in format taken by the pca object
        features = []
        for i,clip in enumerate(detected_clips):
            features.append(clip.features)
        #dimension reduction/ feature extraction with pca  
        pcas = pca.transform(features)
        for i,clip in enumerate(detected_clips):
            clip.pcas = pcas[i]

        return detected_clips

    def spike_sort_self_blur(self, test_data, test_data_startIndex, th, minPeakDist, \
        n_blur, gamma, plotMeanWaveForms):
        """Perform a self blurring stability test
        INPUTS:
            test_data: time series data on which to perform spike sorting
            test_data_startIndex: index from which to start spike sorting
            th: threshold for spike detection
            minPeakDist: minimum sa mples between detected peaks
            n_blur: number of self_blurring operations to be performed
            gamma: self blurring coefficent, >0
            plotMeanWaveforms: bool, is True will plot the waveforms and 
                their means by category
        OUTPUTS:
            stabilities: list of size (n_blur,noCat) of stabilities for each neuron label
                 (cols) for each self_blurring run (rows), up to n_blur runs"""  

        assert gamma > 0, "Self blurring gamma must be greater than 0"

        #Do spike detection on test data time series
        detected_clips = self.spike_detection(test_data, test_data_startIndex, \
            th, minPeakDist=minPeakDist)
        
        #extract features using dwt and PCA
        features_clips = self.feature_extraction(detected_clips, self.params['dwtWavelet'], self.pca)

        #perform classificatiom
        classified_clips, _ = self.classify_SVM(features_clips, self.clf, self.mean_firing_times)
        classified_clips_labels= []
        for clip in classified_clips:
            classified_clips_labels.append([clip.spikeTime, clip.label])

        #compute the mean wavefroms for all classified clips per neuron category
        noCat = self.training_data_params['noCat']
        mean_waveforms = np.zeros((noCat, self.params['clipSize']))
        clipCounts = np.zeros(noCat)
        for clip in classified_clips:
            mean_waveforms[clip.label-1] += clip.data
            clipCounts[clip.label-1] += 1 
        for i,waveform in enumerate(mean_waveforms):
            waveform /= clipCounts[i]
        mean_waveforms
        
        #now create a list of clips for each category type. This is needed since self_blurring
        #only occurs within a category, so it is best to keep them seperate from here on
        classified_by_cat = []
        for i in range(noCat):
            classified_by_cat.append([])
        for clip in classified_clips:
            classified_by_cat[clip.label-1].append(clip)

        # visualise all detected waveforms and their mean
        if plotMeanWaveForms:
            for i,clips in enumerate(classified_by_cat):
                plt.figure(figsize=(2.5,2))
                plt.title(f"Label {i+1}")
                for clip in clips:
                    plt.plot(clip.data, color='black', alpha=0.2)
                plt.plot(self.mean_waveforms[i+1],color='red')
                plt.plot(mean_waveforms[i],color='yellow')
                plt.show()

        #generate clip noises from their mean clips. This is essentially a list of
        #all spike waveforms minus their respective mean waveform
        noise_by_cat = self.gen_noise_by_cat(classified_by_cat, gamma, mean_waveforms)

        #compute the stability of each neuron label n_blur times
        stabilities = []
        for _ in range(n_blur):
            
            #perform self-blurring operation on the clips
            blurred_clips = self.self_blur_random(classified_by_cat, noise_by_cat)

            #no do feature extraction with dwt and PCA
            blurred_features_clips = self.feature_extraction(blurred_clips, \
                self.params['dwtWavelet'], self.pca)
            #classify the features
            classified_blurred, _ = self.classify_SVM(blurred_features_clips, self.clf, \
                self.mean_firing_times)

            #create labels from each clip in the format accepted by the confusion matrix
            #[index, label]
            blurred_labels = []
            for clip in classified_blurred:
                blurred_labels.append([clip.spikeTime, clip.label])

            #create a confuction matrix between the original classification and the
            #self-blurring classification
            qmat, _ = self.confusion_matrix(classified_clips_labels, blurred_labels,\
                self.params['evalTol'], noCat)

            #append the calculated stabilities of each neuron
            stabilities.append(self.stability_analysis(qmat))

        return stabilities

    def gen_noise_by_cat(self, clips_by_cat, gamma, mean_waveforms):
        """Generates a list of noise clips for each clip based on the difference
        between its clip and its respective mean clip, scaled by a value gamma
        INPUTS:
            clips_by_cat: a list of lists, one sublist for each neuron label containing
                all wavefroms of classified clips of that neuron type
            gamma:self-blurring coefficient
            mean_waveforms: a list of the mean waveforms for each neuron type
        OUTPUTS:
            noise_by_cat: a list of same shape as clip_by_cat, but containg 
                the self_blurring noise of each respective clip
            """
        assert gamma > 0, "self_blurring coefficient gamma must be greater than 0"
        noise_by_cat = []
        for i,clips in enumerate(clips_by_cat):
            noise_by_cat.append([])
            for clip in clips:
                noise_by_cat[-1].append(gamma*(clip.data-mean_waveforms[i]))
        return noise_by_cat

    def self_blur_random(self, classified_by_cat, noise_by_cat):
        """Generate a new set of waveforms by adding a randomly chosen self_blurring
        noise clip from a clip in its respective neuron type category
        INPUTS:
            classified_by_cat: a list of lists, one sublist for each neuron label containing
                all wavefroms of classified clips of that neuron type
            noise_by_cat: a list of same shape as clip_by_cat, but containg 
                the self_blurring noise of each respective clip"""
        #for each clip in classified_by_cat, add noise from the randomly shuffled deck
        #of its respective noise list
        self_blurred_clips = []
        for cat, clips in enumerate(classified_by_cat):
            #shuffle the respective noise clips
            shuffled_noise = shuffle(noise_by_cat[cat])
            for i,clip in enumerate(clips):
                #add noise to the clip
                data = clip.data + shuffled_noise[i]
                #create a deep copy, not to alter the original clip, and 
                # append to the output list
                clipCopy = copy.copy(clip)
                clipCopy.data = data
                self_blurred_clips.append(clipCopy)
        return self_blurred_clips

    def eval_new_data(self, data, fs):
        """Apply the spike sorting algorithm to new time series data
        INPUTS:
            data: time series data to be classified
            fs: sampling frequency of the data
        OUTPUTS:
            returns a list of labeled data, each entry of form [index, label]"""
        #filter data
        filtd, th = self.filter(data, fs)
        #spike sorting on test data
        spike_sorting_params = {
            'pca': self.pca, 
            'clf': self.clf,
            'test_data': filtd, 
            'mean_firing_times': self.mean_firing_times,
            'test_data_startIndex': 0, 
            'th': th, 
            'clipSize': self.params['clipSize'], 
            'minPeakDist': self.params['clipDistDef'][0],
            'dwtWavelet': self.params['dwtWavelet']
        }
        return self.spike_sort(**spike_sorting_params)  #return labels

    def self_blur_stability(self, n_blur, gamma, plotMeanWaveforms=False, data=None, fs=None):
        """Perform a self_blurring stability analysis
        INPUTS:
            n_blur: number of times"""
        if data is None:
            data = self.params['sub_data']
            fs = self.params['Fs']
        #filter data
        filtd, th = self.filter(data, fs)
        #do spike sorting on test data with self-blurring analysis
        spike_sorting_params = {
            'test_data': filtd, 
            'test_data_startIndex': 0, 
            'th': th, 
            'minPeakDist': self.params['clipDistDef'][0],
            'n_blur': n_blur,
            'gamma': gamma,
            'plotMeanWaveForms': plotMeanWaveforms
        }
        return self.spike_sort_self_blur(**spike_sorting_params)


#-----------------------------------------------------------------------------------#
#----------------------------------BELOW--------------------------------------------#
#---NOTE: Development functions, left in for backward compatibility purposes only---#
#-----------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------#


def spike_sorter_evaluate(data, hpFreq, Fs, denoisingWavelet, thK, labels, training_data_ratio, \
    clipSize, clipDistDef, dwtWavelet, svmKernel, evalTol, fp_rejection):
    #filter data
    filtd = wdn(data, hpFreq, Fs, wavelet=denoisingWavelet)
    th = thK * np.median(np.abs(filtd))/0.6745

    #choose params to extract training clips from time series
    training_data_params = {
        'data': filtd,
        'labels': labels,
        'training_ratio': training_data_ratio,
        'clipSize': clipSize,
        'noCat': 5,
        'clipDistDef': clipDistDef,
        'clipDetectThreshold': th/4
    }
    training_clips, mean_firing_times, test_data, test_labels, cutoffIndex = \
        gen_training_test_data_spikesorter(**training_data_params)

    if fp_rejection:
        fpClips = extract_fp_clips(filtd, cutoffIndex, training_clips, th/4, clipSize, clipDistDef[0])
        print(f"fp clips detected {len(fpClips)}")
        training_clips.extend(fpClips)

    #perform feature extraction and train SVM 
    pca, clf, training_clips = train_spike_sorter(training_clips, \
        dwtWavelet, svmKernel)

    #spike sorting on test data
    spike_sorting_params = {
        'pca': pca, 
        'clf': clf,
        'test_data': test_data, 
        'mean_firing_times': mean_firing_times,
        'test_data_startIndex': cutoffIndex, 
        'th': th, 
        'clipSize': clipSize, 
        'minPeakDist': clipDistDef[0],
        'dwtWavelet': dwtWavelet
    }
    classified_labels = spike_sort(**spike_sorting_params)

    #create confusonmatrix and evaluate it
    qmat = confusion_matrix(classified_labels, test_labels, evalTol, 5)
    results = evaluate_results_labeled(qmat)
    print(results)
    plot_conf_matrix(qmat)

def gen_training_test_data_spikesorter(data, labels, training_ratio, \
    clipSize, noCat, clipDistDef, clipDetectThreshold):
    assert training_ratio>0 and training_ratio<=1
    cutoff = int(training_ratio * len(data))
    for i in range(len(labels)):
        if labels[i,0] >= cutoff:
            labelIndex = i
            break
    training_data = data[:cutoff]
    training_labels = labels[:labelIndex]
    test_data = data[cutoff:]
    test_labels = labels[labelIndex:]

    training_clips, mean_firing_times, _, _ = \
        extract_labeled_clips(training_data, clipDetectThreshold, training_labels, \
            clipDistDef, clipSize, noCat)
    return training_clips, mean_firing_times, test_data, test_labels, cutoff

def train_spike_sorter(training_clips, dwtWavelet, svmKernel):
    #extract the wavelet coefficients of labeled clips
    training_clips = wavelet_decomposition(training_clips, wavelet=dwtWavelet)
    features = []
    for clip in training_clips:
        features.append(clip.features)

    #extract 5 components
    pca = PCA(n_components=5)
    pcas = pca.fit_transform(features)
    for i,clip in enumerate(training_clips):
        clip.pcas = pcas[i]

    #Train SVM on training clips
    t_data, t_labels = generate_training_data_from_clips(training_clips)
    clf = train_SVM(t_data, t_labels, svmKernel)

    return pca, clf, training_clips

def spike_sort(pca, clf, test_data, mean_firing_times, \
    test_data_startIndex, th, clipSize, minPeakDist, dwtWavelet):
    #Do spike detection on test data time series
    detected_clips = spike_detection(test_data, test_data_startIndex, \
        th, windowSize=clipSize, minPeakDist=minPeakDist)
    #extract wavelet coefficients
    detected_clips = wavelet_decomposition(detected_clips, wavelet=dwtWavelet)
    features = []
    for clip in detected_clips:
        features.append(clip.features)
    #dimension reduction    
    pcas = pca.transform(features)
    for i,clip in enumerate(detected_clips):
        clip.pcas = pcas[i]

    classified_clips = classify_SVM(detected_clips, clf, mean_firing_times)

    #put clips into list of [index,label] for confusion matrix
    classified_clips_labels= []
    for clip in classified_clips:
        classified_clips_labels.append([clip.spikeTime, clip.label])

    return classified_clips_labels

def bandpass__butter_filtfilt(data, fclow, fchigh, fs, order=2):
    """0 phase butterworth bandpass filter
    INPUTS:
        data: data to be filtered
        fclow: lowe cutoff frequency
        fchigh: high cutoff frequency
        fs: sampling frequency of data
        order: order of the butterwoth filer"""
    #generate butterworth sos filter from input params
    # sos = signal.butter(order, [fclow,fchigh], btype="bandpass", fs=fs, output='sos')
    # #return filtered signal usign forward and back filtering for 0 phase filter
    # return signal.sosfiltfilt(sos, data)
    sos = signal.butter(order, [fclow,fchigh], btype="bandpass", fs=fs, output='sos')
    return signal.sosfiltfilt(sos,data)

def hpf(data, fc, fs, order=2):
    sos = signal.butter(order, fc, btype='highpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def lpf(data, fc, fs, order=2):
    sos = signal.butter(order, fc, btype='lowpass', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def lpf_median_filter(data, fc, order, kernelSize, fs):
    """applies a high pass filter followed by a median filter. Also gives an estimated
    threshold value for spike detection based on the low pass filtered time series
    INPUTS:
        data: data to be filteres
        fc: high pass filter cutoff frequency
        order: order of the high pass filter
        kernelSize: size of the median filter kernel, must be odd
        fs: sampling frequency of the input signal
    OUTPUTS:
        filt_data: filtered data
        th_val: calculated threshold value based on high pass filtered data"""

    lpf_data = hpf(data, fc, fs, order=order)
    th = np.median(np.abs(lpf_data))/0.6745     #find threshold based med of abs
    return th, signal.medfilt(lpf_data,kernelSize)

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wdn(data, fclow, fs, wavelet='sym4', level=1, thresh_scaling=0.75):
    """using code from 
    https://www.kaggle.com/theoviel/denoising-with-direct-wavelet-transform"""
    coeffs = pywt.wavedec(data, wavelet, mode='per')
    sigma = (1/0.6745) * madev(coeffs[-level])
    uthresh = thresh_scaling * sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:])
    filt_wvt = pywt.waverec(coeffs, wavelet, mode='per')
    return hpf(filt_wvt, fclow, fs, order=2)




def spike_detection(data, startIndex, threshold, windowSize, minPeakDist):
    """Performs spike detection on the input by thresholding. Steps:
    1. identify spikes by spike detection above a certain threshold and with minimum
    peak distance
    2. extract a window of 64 samples centred around the peak
    3. return a list of Clip objects each containing a clip of a spike
    INPUTS:
        data: data containing the APs
        threshold: thrshold above which an AP is detected
        windowSize: window size around the peak to be taken as a clip
        minPeakdist: minimum distance between peaks in a clip to be 
        considered different
    OUTPUTS:
        clips: a list of clip objects, one for each spike clip
    """
    peaks,_ = signal.find_peaks(data, height=threshold, distance=minPeakDist)
    clips = []
    distFromPeak = int(windowSize/2)
    for peak in peaks:
        start = peak - distFromPeak
        end = peak + distFromPeak 
        peak += startIndex
        clips.append(Clip(peak, data[start:end], [start+startIndex,end+startIndex]))
        if len(clips[-1].data)==0:
            clips.pop(-1)
    return clips


def extract_labeled_clips(data, th, labels, distDef, clipSize, nCat):
    clips = []
    peak = []
    firing_to_peak = []
    for i in range(nCat):
        firing_to_peak.append([])

    for i in range(len(labels)): 
        if i==0:
            #for first, check only distance to second label 
            dist = labels[i+1,0] - labels[i,0]
        elif i==len(labels)-1:
            #check only distance to second to last label
            dist = labels[i,0] - labels[i-1,0]
        else:
            #check label distance ahead
            distBack = labels[i,0] - labels[i-1,0]
            distFront = labels[i+1,0] - labels[i,0]
            #smallest distance is chosen as clip type
            if distBack <= distFront:
                dist = distBack
            else:
                dist = distFront
        
        #now assign the label its clip type
        if dist < distDef[0]:
            clipType = 2
        elif dist < distDef[1]:
            clipType = 1
        else:
            clipType = 0
        
        peaks, _ = signal.find_peaks(data[labels[i,0]:labels[i,0]+distDef[0]], height=th, distance=4)
        if len(peaks) < 1:
            if len(peak)==0 :
                #something went wrong, so dont give it spike time, will asign later 
                # based on average of category. Also label it as merged (since no second spike)
                clips.append(Clip(None, None, None, \
            spikeTime=labels[i,0], label=labels[i,1], seperationType=2))
                continue
            elif peak[-1] < labels[i,0]:
                clips.append(Clip(None, None, None, \
            spikeTime=labels[i,0], label=labels[i,1], seperationType=2))
                continue
            
        else:
            peak.append(peaks[0]+labels[i,0])     #take first peak as peak of that AP

        start = peak[-1] - int(clipSize/2)
        end = peak[-1] + int(clipSize/2)

        clips.append(Clip(peak[-1], data[start:end], [start,end], \
            spikeTime=labels[i,0], label=labels[i,1], seperationType=clipType))
        time_to_peak = clips[-1].peakIndex-clips[-1].spikeTime
        firing_to_peak[clips[-1].label-1].append(time_to_peak)
    
    mean_firing_times = []
    for times in firing_to_peak:
        row = np.array(times)
        mean_firing_times.append(int(np.mean(row)))

    #Go through all clips with unfoudn peaks and find based on categories
    #average firing to peak time
    for clip in clips:
        #check if clip has unsassigned peak time
        if clip.data is None:
            clip.peakIndex = clip.spikeTime + mean_firing_times[clip.label-1]
            start = clip.peakIndex - int(clipSize/2)
            end = clip.peakIndex + int(clipSize/2)
            clip.data = data[start:end]
            clip.indexRange = [start,end]

    false_clips = 0
    correct_clips = []
    for clip in clips:
        pred_firing_time = clip.peakIndex - mean_firing_times[clip.label-1]
        if np.abs(pred_firing_time - clip.spikeTime) > 10:
            false_clips += 1
        else:
            correct_clips.append(clip)

    total = len(correct_clips)
    merged = 0
    connected = 0
    isolated = 0
    for clip in correct_clips:
        if clip.seperationType == 2:
            merged += 1
        elif clip.seperationType == 1:
            connected += 1
        elif clip.seperationType == 0:
            isolated += 1

    return correct_clips, mean_firing_times, false_clips, (isolated,connected,merged)

def extract_fp_clips(data, cutoffIndex, labeled_clips, th, windowSize, minPeakDist):
    #detect peaks using spike detector
    clips = spike_detection(data[0:cutoffIndex], 0, th, windowSize, minPeakDist)
    for clip in clips:
        clip.spikeTime = clip.peakIndex - 10    #roughly time between peak and spike
    #go through each clip, if not within 10 of any labeled data, it is fp
    fp_clips = []
    for clip in clips:
        goodClip = False
        for lclip in labeled_clips:
            if np.abs(lclip.spikeTime - clip.spikeTime)<10:
                goodClip = True
                continue
        if not goodClip:
            clip.label = 0
            fp_clips.append(clip)
    return fp_clips


def wavelet_decomposition(clips, wavelet='sym4'):
    #compute maximum useful wavelet decomposition level
    decomp_level = pywt.dwt_max_level(len(clips[0].data), wavelet)
    dwt_clips = []
    for clip in clips:
        coeffs = pywt.wavedec(clip.data, wavelet, level=decomp_level)
        coeffs_flat = []
        for row in coeffs:
            coeffs_flat.extend(row)
        coeffs = np.array(coeffs_flat)
        clip.features = coeffs
    return clips

def get_shuffled_train_test_data(clips, train_to_test_ratio):
    assert train_to_test_ratio>0 and train_to_test_ratio<1

    shuffled_clips = shuffle(clips)
    data = []
    labels = []
    clip_type = []
    for clip in shuffled_clips:
        data.append(clip.pcas)
        labels.append(clip.label)
        clip_type.append(clip.seperationType)

    cutoff = int(train_to_test_ratio*len(data))
    training_data = data[0:cutoff]
    training_labels = labels[0:cutoff]
    test_data = data[cutoff:]
    test_labels = labels[cutoff:]
    test_clip_types = clip_type[cutoff:]
    return (training_data, training_labels, test_data, test_labels, test_clip_types)

def generate_training_data_from_clips(clips):
    data = []
    labels = []
    for clip in clips:
        data.append(clip.pcas)
        labels.append(clip.label)   
    return data,labels

def train_SVM(training_data, training_labels, svmKernel):
    #train SVM
    clf = svm.SVC(probability=True, kernel=svmKernel)
    clf.fit(training_data, training_labels)
    #find average distance from peak to firing index for each neuron type

    return clf

def eval_SVM(data_sorted):
    clf = svm.SVC()
    clf.fit(data_sorted[0], data_sorted[1])
    predictions = clf.predict(data_sorted[2])  

    test_labels = data_sorted[3]
    test_clip_types = np.array(data_sorted[4])

    correct_types = np.zeros(4)
    total_types = np.zeros(4)
    total_types[0] = np.sum(test_clip_types==0)
    total_types[1] = np.sum(test_clip_types==1)
    total_types[2] = np.sum(test_clip_types==2)
    total_types[3] = np.sum(total_types[:-1])   #total
    accuracies_types = np.zeros(4)
    for i,label in enumerate(predictions):
        if label == test_labels[i]:
            correct_types[test_clip_types[i]] += 1
    correct_types[-1] = np.sum([correct_types[:-1]])

    accuracies_types = correct_types/total_types
    return accuracies_types

def eval_SVM_withoutClipType(data_sorted):
    clf = svm.SVC()
    clf.fit(data_sorted[0], data_sorted[1])
    predictions = clf.predict(data_sorted[2])  
    test_labels = data_sorted[3]
    correct = 0
    for i,label in enumerate(predictions):
        if label == test_labels[i]:
            correct += 1
    return correct/len(test_labels) 


def confusion_matrix(classified_1, classified_2, sample_tol, no_categories):
    """creates a confusion matrix between 2 sets of classification
    INPUTS:
        classified_1,2 = array of classified data with each entry as columns
            row 0: index number
            row 1: category assigned to the data entry
        sample_tol: sample tolerance for 2 data entries of classified_1 and
                    classified_2 to be considered the same
        no_categories: number of categories used for classification
    OUPUTS:
        returns Qmat, the confusion matrix"""
    c1 = np.array(classified_1)
    c2 = np.array(classified_2)
    #to track if entries in classified 2 have been associated 
    c2_associated = np.zeros(c2.shape[0])   

    #create confusion matrix
    Qmat = np.zeros([no_categories+1, no_categories+1])

    #Go through c1, associate all c2 within tolerance, and fill in confusion matrix
    for entry in c1:
        c1_index = entry[0]
        c1_category = entry[1]
        associated = np.where((c2[:,0]>=c1_index-sample_tol) \
            & ((c2[:,0]<=c1_index+sample_tol))) 
        associated = associated[0]  #take c1_index samples

        #no associated points found in c2, add to null column for c2
        if len(associated)<1 :
            Qmat[c1_category-1,-1] += 1
        else:
            #if just one associated found, then associate them
            if len(associated) == 1:
                c2_index = associated[0]
            #check if multiple associated spikes were detected, and choose closest
            elif len(associated)>1:
                c2_index = associated[0]
                ds = np.abs(c1_index - c2[c2_index,0])
                for i in associated[1:]:
                    if np.abs(c1_index - c2[i,0]) < ds:
                        c2_index = i
                        ds = np.abs(c1_index - c2[i,0])
            #Check if associated c2 has alredy been associated, if not associate and 
            #fill the confusion matrix
            if c2_associated[c2_index] == 0:
                c2_associated[c2_index] = 1
                c2_category = c2[c2_index,1]
                Qmat[c1_category-1,c2_category-1] += 1
    
    #go through c2 and find all non-associated entries and add in null row of c1
    for i,asso in enumerate(c2_associated):
        #check if not been associated
        if asso == 0:
            index = i
            c2_category = c2[i,1]
            #place in confusion matrix
            Qmat[-1,c2_category-1] += 1

    return Qmat

def stability_analysis(Qmat):
    """Performs a stability analysis of each category between two classifications
    INPUTS:
        Qmat: confusion matrix of two different classifications
    OUTPUTS:
        stabilities: array of computed stability for each category"""
    stabilities = np.zeros(Qmat.shape[0]-1)
    #compute stability of each category (expluding nulls)
    for k in range(Qmat.shape[0]-1):
        stabilities[k] = 2 * Qmat[k,k] / (np.sum(Qmat[k,:]) + np.sum(Qmat[:,k]))
    return stabilities



#-------------------------------For labeled data--------------------------------------
def evaluate_results_labeled(Qmat, fp_removed):
    """function to evaluate false positives, false negatives, and ratio of 
    corretly labeled spikes per neuron category
    INPUTS:
        Qmat: confusion matrix using as classification 1 the labeled data and 
            classification 2 as the results of the classifier under test 
    OUTPUTS:
        results: labeled dict containg the evaluation parameters calculated from 
            the confusion matrix
            ->"fp": false positives, identified spikes with used classifier but not 
                found in labeled data
            ->"fn": false negatives, identified spikes in labeled data not identified
                in the classified data
            ->"ta": total accuracy, the sum of correctly classified spikes (diagonal
                values excluding null row and column) divided by the sum of 
                the all entries in the confusion matrix
                """

    false_pos = np.sum(Qmat[:,-1])
    false_neg = np.sum(Qmat[-1,:])
    trueClips = Qmat[:-1,:-1]
    correct_classified = np.multiply(trueClips, np.eye(len(trueClips))).sum()
    miss_classified = trueClips.sum() - correct_classified
    total_accuracy =  correct_classified / np.sum(Qmat)
    return {
        'false_pos': false_pos,
        'false_pos_removed': fp_removed,
        'false_neg': false_neg,
        'correct_classified': correct_classified,
        'miss_classified': miss_classified,
        'total_accuracy': total_accuracy
    }

def plot_conf_matrix(qmat):
    #plot the confusion matrix
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
