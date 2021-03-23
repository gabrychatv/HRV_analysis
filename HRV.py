import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hrvanalysis import get_frequency_domain_features, get_time_domain_features
"""
import pyhrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl
"""
from scipy.integrate import trapz
from scipy import signal
from scipy.interpolate import interp1d

#read data
data = pd.read_csv('signals/ok/1.txt', header=None)[0]
time = np.cumsum(data/1000/60)
#print(data.head())

#plot tachogram
plt.plot(time, data)
plt.title('Tachogram')
plt.xlabel('Time [min]')
plt.ylabel('RR Intervals [ms]')
#plt.show()
#pyhrv.tools.tachogram(data)

#plot Poincare
plt.plot(data[0:len(data)-1], data[1:len(data)], 'r.', markersize=4)
plt.title('Poincare Plot')
plt.xlabel('RR Intervals [i]')
plt.ylabel('RR Intervals [i+1]')
#plt.show()
#plot Poincare pyhrv
#nl.poincare(data)

#funkcje
def time_domain(rr):
    #MEAN RR
    mean = sum(rr) / len(rr)
    print('Mean RR = ' + str(round(mean, 0)) + ' ms')
    #STD RR (sdnn)
    sdnn = np.sqrt(sum((mean - rr)**2) / (len(rr) - 1))
    print('SDNN / STD RR = ' + str(round(sdnn, 2)) + ' ms')
    #RMSSD
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    print('RMSSD = ' + str(round(rmssd, 2)) + ' ms')
    #pNN50
    a = []
    for i in range (0, len(data)-1):
        x = np.abs(data[i+1]-data[i])
        if x > 50:
            NN50 = x
            a.append(NN50)
    pNN50 = len(a)*100/i
    print('pNN50 = ' + str(round(pNN50, 1)) +'%')

def frequency_domain(rr, fs = 4):
    # Estimate the spectral density using Welch's method
    fxx, pxx = signal.welch(x = rr, fs = fs)
   
    cond_vlf = (fxx >= 0) & (fxx < 0.04)
    cond_lf = (fxx >= 0.04) & (fxx < 0.15)
    cond_hf = (fxx >= 0.15) & (fxx < 0.4)
    
    # calculate power in each band by integrating the spectral density 
    vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
    lf = trapz(pxx[cond_lf], fxx[cond_lf])
    hf = trapz(pxx[cond_hf], fxx[cond_hf])

    total_power = vlf + lf + hf

    # fraction of lf and hf
    lf_nu = 100 * lf / (lf + hf)
    hf_nu = 100 * hf / (lf + hf)

    #HF_nu
    print("HFnu = " + str(round(hf_nu, 3)))
   
    #LF/HF
    print("LF/HF = " + str(round(lf/hf, 3)))

time_domain(data)
frequency_domain(data)

#times = get_time_domain_features(data)
#print(times)
#freq = get_frequency_domain_features(list(data))
#print(freq)
#print(fd.welch_psd(data))

