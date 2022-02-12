import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.signal as signal
import numpy as np
import ss 
import copy

def bandpass_filt(data, fclow, fchigh, fs, order=4):
    sos = signal.butter(order, [fclow,fchigh], btype="bandpass", fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

#get data
mat = spio.loadmat('data/submission.mat', squeeze_me=True) 
d = mat['d']
# Index = mat['Index'] 
# Class = mat['Class']
# real_class = []
# for i in range(len(Index)):
#     real_class.append([Index[i],Class[i]])
# real_class = np.array(sorted(real_class, key=lambda x: x[0]))

filtd = ss.wdn(d, 50, 25e3, level=1)
th = 4 * np.median(np.abs(filtd))/0.6745
clips = ss.spike_detection(filtd, th, 64, 15)
print(len(clips))
plt.title("Detected Spikes in Submission Data")
plt.plot(filtd[3400:4000])
plt.xlabel("Sample no.")
plt.ylabel("Voltage [mv]")
plt.axhline(th,linestyle='-',color='black', alpha=0.5)
plt.text(-25, th+0.1, "Threshold")
plotfirst = True
for i,clip in enumerate(clips):
    if clip.peakIndex > 4000:
        break
    if clip.peakIndex<3400:
        continue
    if plotfirst:
        plt.axvspan(clip.indexRange[0]-3400, clip.indexRange[1]-3400, color='green', alpha=0.4)
        plotfirst = False
    plt.axvline(clip.peakIndex-3400, color='black')
plt.tight_layout()
plt.show()




pass