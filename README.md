# SpikeSorting-WaveletSVM
A spike sorting algorithm for classifying neuron action potential (off-line). The algorithm uses the discrete wavelet transfrom followed by PCA as a new basis for the time series data. This optimises the information content between the time and frequency domains to better distinguish overlapping action poentials. A support vector machine (SVM) is trained on simulated data and classifies the neuron action potentials. It also incorporates false positive detection to reject spurious spikes.

The classifier was evaluated on an unlabled real neuronal recording from an identical part of the brain. Because of the lack of ground-truth for the real data, the model was evaluated using self-blurring and stability whereby noise is injected in the post-processed data and the stability of the classification is measured. The more a classification remains unchanged (stable) with injected noise, the more likely it is to be using the correct basis for classification.  

<p align="center">
  <img src="https://user-images.githubusercontent.com/60844959/153716675-36aa0b9f-1c5e-4ed3-a545-0d7d5ef2eaaf.png" />
</p>

