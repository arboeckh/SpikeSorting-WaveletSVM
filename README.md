# SpikeSorting-WaveletSVM
A spike sorting algorithm for classifying neuron action potential (off-line). The algorithm uses the discrete wavelet transfrom followed by PCA as a new basis for the time series data. This optimises the information content between the time and frequency domains to better distinguish overlapping action poentials. A support vector machine (SVM) is trained on simulated data and classifies the neuron action potentials. It also incorporates false positive detection to reject spurious spikes.

<p align="center">
  <img src="https://user-images.githubusercontent.com/60844959/153716675-36aa0b9f-1c5e-4ed3-a545-0d7d5ef2eaaf.png" />
</p>

![image](https://user-images.githubusercontent.com/60844959/153716916-d4539f0a-c281-419b-b297-96068cba57fc.png)
