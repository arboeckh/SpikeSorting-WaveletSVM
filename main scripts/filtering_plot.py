"""Script to compare a bandpass filter to wavelet denoising for the submission data, 
and its effect the sensitivity of subsequent spike detection to the detection threshold"""

import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np
import ss 

#get training and submission data
mat = spio.loadmat('data/training.mat', squeeze_me=True) 
d = mat['d']
Index = mat['Index'] 
mat_s = spio.loadmat('data/submission.mat', squeeze_me=True) 
ds = mat_s['d']

#filter with 50 hp filter
filt_org = ss.hpf(ds, 50, 25e3)
#also with 300-3kHz bandpass
filt_bp_s = ss.bandpass__butter_filtfilt(ds, 300, 3000, 25e3)
#finally with wavelet denoising
filt_wvt_s = ss.wdn(ds, 50, 25e3, level=1)

spikecount = len(Index)

#find the number of spikes detected for bp and wdn filters for various values of threshold
ths_sub = np.linspace(0.01, 3, 20)
bp_spikes_sub = []
bp_stab_sub = None
wvt_spikes_sub = []
wvt_stab_sub = None
for i,th in enumerate(ths_sub):

    bp_spikes_sub.append(len(ss.spike_detection(filt_bp_s, 0, th, 128, 20)))
    #if now detecting less spikes than the estimated spike count, calculate the slope at this point
    if bp_spikes_sub[-1] - spikecount < 0:
        if bp_stab_sub is None:
            #find slope and store the index at which it occurs for plotting
            bp_stab_sub = (bp_spikes_sub[-1]-bp_spikes_sub[-2] ) / (ths_sub[i]-ths_sub[i-1])
            bp_stab_index_sub = i

    wvt_spikes_sub.append(len(ss.spike_detection(filt_wvt_s, 0, th, 128, 20)))
    #same as above, but for wdn
    if wvt_spikes_sub[-1] - spikecount < 0:
        if wvt_stab_sub is None:
            wvt_stab_sub = (wvt_spikes_sub[-1] - wvt_spikes_sub[-2]) / (ths_sub[i]-ths_sub[i-1])
            wvt_stab_index_sub = i

#ploting
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3,1]})

#show a plot with filtered waveforms for all 3 filter types
plt.subplot(2,1,1)
plt.plot(filt_org, linewidth=1)
plt.plot(filt_bp_s, linewidth=2)
plt.plot(filt_wvt_s, linewidth=2)
plt.xlim([526200,526800])
plt.ylim([-2,4])
plt.xlabel("Sample no.")
plt.ylabel("Voltage [mV]")
plt.title("Comparison of band-pass and wavelet denoising \non submission data")
plt.legend(["50Hz high-pass","0.3-3kHz band-pass","level 1 wavelet denoising"])

#plot the number of spikes detected vs. threshold for bp and wdn filters
plt.subplot(2,1,2)
plt.title("Detected spikes in submission data")
plt.xlabel("Threshold value")
plt.ylabel("Number of detected spikes")
plt.ylim([0,30000])
#plot bp and wdn spikes detected vs th
plt.plot(ths_sub, bp_spikes_sub,'+', color='blue')
plt.plot(ths_sub, wvt_spikes_sub, '.', color='orange')
#plot slope and print it on plot for bp
plt.plot(ths_sub[bp_stab_index_sub-1:bp_stab_index_sub+1], bp_spikes_sub[bp_stab_index_sub-1:bp_stab_index_sub+1], color='black')
plt.text(ths_sub[bp_stab_index_sub-1], spikecount +1000, f"m={int(bp_stab_sub)}")
#same but for wdn
plt.plot(ths_sub[wvt_stab_index_sub-1:wvt_stab_index_sub+1], wvt_spikes_sub[wvt_stab_index_sub-1:wvt_stab_index_sub+1], color='black')
plt.text(ths_sub[wvt_stab_index_sub-1], spikecount +1000, f"m={int(wvt_stab_sub)}")
plt.axhline(spikecount, color='grey', alpha=0.5)
plt.legend(["300-3kHz bandpass","wavelet denoising"])

plt.tight_layout()
plt.show()







pass