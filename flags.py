import numpy as np
import mne
import os
import matplotlib.pyplot as plt                    
import argparse
import easygui
import math
import xlsxwriter  # pip install XlsxWriter

from datetime import datetime    
from tkinter import messagebox
from scipy import signal, stats
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
import timeit

#Print the system information
mne.sys_info()

def file_name(path):
    name=os.path.splitext(os.path.basename(path))[0]
    if '_2020' in name:
        ind = str.index(name,'_2020')
        name=name[:ind]

    name=name + datetime.now().strftime("_%Y%B%d_%H-%M") 
    return name

def load_brainvision_vhdr(path):
    # Import the BrainVision data into an MNE Raw object
    mne.set_log_level("WARNING")
    print('Reading raw file...')
    print('')
    raw= mne.io.read_raw_brainvision(path, 
            preload=True, 
            verbose=True)
    raw.rename_channels(lambda s: s.strip("."))

    return raw    

def show_info(raw):  #Brainvision files
    raw.rename_channels(lambda s: s.strip("."))    # strip channel names of "." characters
    print()
    print('------------------------------ Show Info -----------------------------')
    print('File:', __file__)
    print('')
    _, times = raw[:, :] 
    print('Data type: {}\n\n{}\n'.format(type(raw), raw))
    # Give the size of the data matrix
    print('%s channels x %s samples' % (len(raw.info['ch_names']), len(raw.times)))
    # Give the sample rate
    print('Sample rate:', raw.info['sfreq'], 'Hz')

def set_sleep_stages(raw,path_stages): #Colored sleep stages
    stages= np.loadtxt(path_stages,delimiter =' ', usecols =(0) )
    n_anot= stages.shape[0]
    epoch_length=30
    onset = np.zeros((n_anot))        
    duration = np.zeros((n_anot))    
    description = np.zeros((n_anot))  
    start=0

    for i in range(n_anot):
        onset[i] = start
        duration[i] = epoch_length 
        description[i] = stages[i]
        start= start + epoch_length
    
    stages_anot= mne.Annotations(onset,duration,description, orig_time=raw.annotations.orig_time)    
    reraw = raw.copy().set_annotations(stages_anot)

    return reraw, stages_anot

def extract_data(raw,name_channel): #Extracting data, sampling frequency and number of samples
    data, sf, chan =raw._data, raw.info['sfreq'], raw.info['ch_names']   
    n = data.shape[1]   # Number of samples  
    channel = (raw.ch_names).index(name_channel)
    data=data[channel][:]    
    data=data.tolist() # Data (type = list)
    
    return data, sf, n

def LPF(senal, low_pass = 'pad'):  #Low pass filter (Butterworth - Fc = 30 Hz - Order = 5)
    order = 5
    sampling_freq = 200
    cutoff_freq = 30
    sampling_duration = 30
    number_of_samples = sampling_freq * sampling_duration
    time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

    normalized_cutoff_freq = 2*30/sampling_freq #Normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    order_filter=5

    b, a  = signal.butter(order_filter, normalized_cutoff_freq, output ='ba')  #Return two arrays: numerator and denominator coefficients of the filter.

   
    filtered_signal = signal.filtfilt(b, a, senal, method = low_pass)
    
    return filtered_signal

def BP(senal, notpad = True, band_pass = 'constant'): #Band pass filter (Chebyshev II - Fc1 = 0.125 Hz - Fc2 = 6 Hz - Order = 16)
    order = 16
    rs= 40   #The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
    Wn= [0.125, 6]   #Critical frequencies (For type II filters, this is the point in the transition band at which the gain first reaches -rs. 
    #For digital filters, Wn are in the same units as fs. Wn is in half-cycles / sample.

    sos = signal.cheby2(order, rs, Wn, btype='bandpass', analog=False, output='sos', fs=100)
    
    if notpad:
        filtered = signal.sosfiltfilt(sos, senal) # default = no pad
    else:
        filtered = signal.sosfiltfilt(sos, senal, padtype = band_pass)

    return filtered

def filtering(raw,name_channel,sf, overlapping = 50, win = 5, bandpass_ = 'even', lowpass_ = 'gust'): #Filering with windows of five seconds and selected overlapping 
    # Overlapping = 1 to 99 --> average of previous filtering and the new one

    data, sf, n = extract_data(raw, name_channel)
    pos = 0
    data_out = np.zeros([1,len(data)])
    #Signal - avg(signal)
    data_referenced =  data - np.mean(data)

    while pos < n - 1 : #pos is the last element of the window 
        
        if pos == 0:
            vec = data_referenced[0:int(win*sf)] # 5 secs of the signal

            # 1° Downsampling 200 Hz --> 100 Hz
            data_downsampled = signal.decimate(vec, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
            # 2° Low-pass Fc = 30 Hz
            data_LP = LPF(data_downsampled, low_pass=lowpass_)
            # 3° Band-pass Cheby II 0.5-4 Hz
            x = BP(data_LP, notpad=False, band_pass = bandpass_)
            # 4° Upsampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne 
            dat = signal.resample(x, len(vec), t=None, axis=0, window=None) 
            
            data_out[0,0:int(win*sf)] = dat

        else:

            if pos + sf * win < n-1:  #Each 5 second window instead of the last one
           
                vec = data_referenced[pos:pos+int(win*sf)]

                # 1° Downsampling 200 Hz --> 100 Hz
                data_downsampled = signal.decimate(vec, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
                # 2° Low-pass Fc = 30 Hz
                data_LP = LPF(data_downsampled, low_pass=lowpass_)
                # 3° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, notpad=False, band_pass = bandpass_)
                # 4° Upsampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne 
                dat = signal.resample(x, len(vec), t=None, axis=0, window=None) 

                
                v1 = dat[0:int(win*sf*overlapping/100)]
                v2 = data_out[0,pos:int(pos+win*sf*overlapping/100)]
                w = np.linspace(0,1,len(v1))
                aux1 = (v1*w+v2*(1-w))
                aux2 = dat[int(win*sf*overlapping/100):]

                aux = np.concatenate((aux1, aux2))

                data_out[0,pos:pos+int(win*sf)] = aux

            else:   #The last part of the signal to filter (which could be less than 5 seconds)

                vector = data[pos:]

                # 1° Downsampling 200 Hz --> 100 Hz
                data_downsampled = signal.decimate(vec, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
                # 3° Low-pass Fc = 30 Hz
                data_LP = LPF(data_downsampled, low_pass = lowpass_)
                # 4° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, notpad=False, band_pass = bandpass_)
                # 5° Oversampling 100 Hz --> 200 Hz   //// In order to plot this signal, otherwise it can't be plotted with mne tools
                dat = signal.resample(x, len(vector), t=None, axis=0, window=None) 

                aux1 = (dat[0:int(win*sf*overlapping/100)] + data_out[0,pos:int(pos+win*sf*overlapping/100)])/2
                aux2 = dat[int(win*sf*overlapping/100):]

                aux = np.concatenate((aux1, aux2))

                data_out[0,pos:] = aux 

        pos = int(pos + win*sf - overlapping/100*sf*win)

    return data_out

def extract_td(x):  #Extract the position and duration of the labeled 'KC'    
    vector = np.zeros([1,2])

    for i in range(len(x)-1):
        cantKC = 0
        if x[i] == ',' and x[i+1] != 'K':

            start =  float(x[0:i])
            vector[0,0] = start           
            duration = float(x[i+1:len(x)-4])
            vector[0,1] = duration
            cantKC += 1

    return vector, cantKC

def pos_firstcomma(x):
    for i in range(len(x)-1):
        if x[i] == ',' and (x[i+1] != 'K' and i != len(x)- 5):  
            pos = i 
    return pos

def pos_seccomma(x):
    for i in range(len(x)-1):
        if x[i] == ',' and (x[i+1] == 'K' or i == len(x)-5):  
            pos = i 
    return pos

def extract_td_phase2(x):  #Extract the position and duration of the signal with phase 2 scoring   
    vector = np.zeros([1,2])  
    for i in range(len(x)-1):  
        cant_p2 = 0
        if x[i] == ',' and x[i+1] != '2':
            start =  float(x[0:i])
            vector[0,0] = start           
            duration = float(x[i+1:len(x)-5])
            vector[0,1] = duration
            cant_p2 += 1
    
    return vector, cant_p2

def KCs(raw,name_channel, sf, path_KC): #Return a vector with only KCs labeled (everything else is zero)
    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])

    filetxt = open(path_KC, 'r')

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = data[start:end]
    filetxt.close()
    return data_out

def pulse(time_shape,sfreq): #To generate 0.5 sec x-axes
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    # Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # Pulse signal
    return pulso

def detect_if_S2(path_KC): #Return a list with all stages with phase 2 labeling (together) 
    starts_S2 =[]
    filetxt = open(path_KC, 'r')
    for x in filetxt:
        if x[len(x)-4:len(x)-1] == '2.0':
            vector, cantKC = extract_td_phase2(x)  #vector[0,0] = position // vector[0,1] = duration
            start = int(vector[0,0])
            starts_S2.append(start)

    filetxt.close()

    return starts_S2

def normalize_flags(flags):
    posiciones = np.where(flags)
    flags_out = np.zeros(flags.shape[0])

    for i in posiciones:
        flags_out[i] = 25e-6

    return flags_out

def put_flag(filt, sf = 200, name_channel = 'C4_1', path_KC = ''):
    
    x = filt[0]
    tamano = x.shape[0]
    flags = np.zeros(tamano)
    window = 2
    if_S2 = detect_if_S2(path_KC)
    print(len(if_S2))
    print(tamano//(30*sf))

    for start in if_S2:
        inicio = int(start*sf)
        final = inicio + int(30*sf)
        i = inicio
        while i < final:
            data = x[i:int(i+window*sf)].tolist()
            maxi = max(data)
            mini = min(data)
            pos_maxi = data.index(maxi)
            pos_mini = data.index(mini)
            maxi = maxi * (10**6)
            mini = mini * (10**6)
            a_pp = (maxi - mini)
            t_mini_maxi = (pos_maxi - pos_mini)/sf


            if (pos_mini<pos_maxi) and (a_pp>75) and (maxi>20) and (mini<-30) and (t_mini_maxi<0.9) and (t_mini_maxi>0.14):
                c = (pos_maxi + pos_mini) // 2
                flags[i+int(c-sf):i+int(c+sf)] = 25e-6
                i = i + int(window*sf*0.25)
                print('PULSO')
            else:
                i = i + int(window*sf*0.25)
            
            print(i/sf, 'of', tamano/sf)
    
    #flags_out = normalize_flags(flags)

    return flags

def time_stages(raw,name_channel, sf, filetxt): 
    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = data[start:end]

    return data_out

def re_esctructure(raw, path_KC): #Re-estructure data
    data, sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]

    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    filtered_signal = filtering(raw, 'C4_1', 200)
    pulse_ch = pulse(time_shape,sfreq)

    new_data=data.copy()
    new_data[0]= pulse_ch
    new_data[1]= c4_1 
    new_data[2]= filtered_signal
    new_data[3]= KCs(raw, 'C4_1', 200, path_KC)
    new_data[4]= put_flag(filtered_signal, path_KC = path_KC)
    new_data[5]=  pulse_ch
    
    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['Pulse-0', 'EEG', 'Filt', 'KCs', 'Flags', 'Pulse-1']
    new_chtypes = ['misc'] + 4*['eeg'] + ['misc'] # Remake channels
    
    # Initialize an info structure      
    new_info = mne.create_info(new_ch_names, sfreq=sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']      # Record timestamp for annotations
    
    new_raw=mne.io.RawArray(new_data, new_info)        # Build a new raw object 
    new_raw.set_annotations(raw.annotations)         
    
    return new_raw, filtered_signal

def plot(raw,n_channels,scal,order): 
    """To visualize the data"""
    raw.plot(show_options=True,
    title='Etiquetado',
    start=0,                        # Initial time to show
    duration=30,                    # Time window (sec) to plot in a given time
    n_channels=n_channels, 
    scalings=scal,                  # Scaling factor for traces.
    block=True,
    order=order)

############# Main function ##########################################################
def main():  # Wrapper function
    messagebox.showinfo(message ="This program allows you to put flags.", title="Info")

    path = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS40.vhdr' #easygui.fileopenbox(title ='Select VHDR file.')
    raw = load_brainvision_vhdr(path)
    show_info(raw)
    path_states = 'C:/Users/aylin/PFC/Señales/TXT/ExpS40.txt' #easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') # To select txt file of previous anotations
    raw,_ = set_sleep_stages(raw,path_states)
    
    path_KC = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS40_2020September29_11-10.txt' #easygui.fileopenbox(title ='Select .txt file with KCs positions and durations.')
    ### ExpS11_2020September28_19-48.txt
    ### ExpS12_2020September28_20-01.txt
    ### ExpS17_2020September28_20-14.txt
    ### ExpS21_2020September28_20-29.txt
    ### ExpS28_2020September28_20-40.txt
    ### ExpS36_2020September29_14-00.txt
    ### ExpS37_2020September29_11-01.txt
    ### ExpS40_2020September29_11-10.txt

    raw_end, filtered_signal = re_esctructure(raw, path_KC)
    
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3, stim=2)
    n_channels = 6
    order=[0,1,2,3,4,5]
    plot(raw_end,n_channels,scal,order)
    


if __name__ == '__main__':
    main()