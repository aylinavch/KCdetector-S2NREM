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
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


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

def LPF(senal, notpad = True):  #Low pass filter (Butterworth - Fc = 30 Hz - Order = 5)
    order = 5
    sampling_freq = 200
    cutoff_freq = 30
    sampling_duration = 30
    number_of_samples = sampling_freq * sampling_duration
    time = np.linspace(0, sampling_duration, number_of_samples, endpoint=False)

    normalized_cutoff_freq = 2*30/sampling_freq #Normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    order_filter=5

    b, a  = signal.butter(order_filter, normalized_cutoff_freq, output ='ba')  #Return two arrays: numerator and denominator coefficients of the filter.

    if notpad:
        filtered_signal = signal.filtfilt(b, a, senal)
    else:
        filtered_signal = signal.filtfilt(b, a, senal, method='gust')
    
    return filtered_signal

def BP(senal, notpad = True): #Band pass filter (Chebyshev II - Fc1 = 0.125 Hz - Fc2 = 6 Hz - Order = 16)
    order = 16
    rs= 40   #The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
    Wn= [0.125, 6]   #Critical frequencies (For type II filters, this is the point in the transition band at which the gain first reaches -rs. 
    #For digital filters, Wn are in the same units as fs. Wn is in half-cycles / sample.

    sos = signal.cheby2(order, rs, Wn, btype='bandpass', analog=False, output='sos', fs=100)
    
    if notpad:
        filtered = signal.sosfiltfilt(sos, senal)
    else:
        filtered = signal.sosfiltfilt(sos, senal, padtype='constant')

    return filtered

def filtering(raw,name_channel,sf, overlapping = 50, win = 5): #Filering with windows of five seconds and selected overlapping
    
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
            data_LP = LPF(data_downsampled, notpad=False)
            # 3° Band-pass Cheby II 0.5-4 Hz
            x = BP(data_LP, notpad=False)
            # 4° Upsampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne 
            dat = signal.resample(x, len(vec), t=None, axis=0, window=None) 
            
            data_out[0,0:int(win*sf)] = dat

        else:

            if pos + sf * win < n-1:  #Each 5 second window instead of the last one
           
                vec = data_referenced[pos:pos+int(win*sf)]

                # 1° Downsampling 200 Hz --> 100 Hz
                data_downsampled = signal.decimate(vec, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
                # 2° Low-pass Fc = 30 Hz
                data_LP = LPF(data_downsampled,notpad=False)
                # 3° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, notpad=False)
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
                data_LP = LPF(data_downsampled,notpad=False)
                # 4° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, notpad=False)
                # 5° Upsampling 100 Hz --> 200 Hz   //// In order to plot this signal, otherwise it can't be plotted with mne tools
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

def center(KC):
    maxi = max(KC)
    maxi_pos = KC.index(maxi)
    mini = min(KC)
    mini_pos = KC.index(mini)

    c = (maxi_pos + mini_pos) // 2

    return c

def align_KC(start, end, data, sf):
    d = 2 #in seconds
    dur = d * sf
    KC_numpy = data[start:end]
    KC = KC_numpy.tolist()
    c = center(KC)
    cent = c + start
    mid_dur = (2 - (start - end)) // 2
    starting = cent - int(sf*d/2)
    ending = starting + d * sf

    a_KC = data[int(starting):int(ending)]


    return a_KC, len(a_KC)

def getKCs(raw,name_channel, sf, path_KC): #Return a vector with only KCs labeled (everything else is zero)
    filetxt = open(path_KC, "r")
    filt = filtering(raw, name_channel,sf)
    timeseries = []
    data = filt[0]

    for x in filetxt:
        if x[-3] == 'K' and x[-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            a_KC, leng = align_KC(start, end, data, sf)
            timeseries.append(a_KC.tolist())
      
    filetxt.close()
    return timeseries

def get_noKCs(raw, name_channel, sf, sN = 36):
    filt = filtering(raw, name_channel,sf)
    timeseries = []
    data = filt[0]
    noKCs = []
    
    if sN == 12:
        pos_inic_noKC = [1005,1180,1220,1320,1360,1485,1585,1645,1730,1860,1945,2020,2130,2215]
    elif sN == 17:
        pos_inic_noKC = [910,920,995,1050,1140,1175,1235,1275,1330,1375,1445,1525,1565,1660,1730,2033]
    elif sN == 21:
        pos_inic_noKC = [575,605,690,740,805,845,920,955,1015,1090,1160,1200,1255,1260,2715,2770,2805,2860,2875,2905]
    elif sN == 36:
        pos_inic_noKC = [435,500,620,750,820.5,1065,1180,1270,1340,1510,1660,1825,1900,1980,2018,2100,2130,2180,2420,2600,3035,3350,3540,3665,3875,4100]
    elif sN == 40:
        pos_inic_noKC = [755,925,945,1040,1090,1160,1200,1255,1295,1370,1495,1840]
    else:
        print('Do not use S',sN, ' because there are no KCs enough to make a pipeline')


    for start in pos_inic_noKC:
        noKCs.append(data[int(start):int(start+2*sf)].tolist())

    return noKCs

def clasification(raw,name_channel,sf,path_KC, s_N):
    #preprocessing.StandardScaler
    #preprocessing.scale
    
    yesKCs = getKCs(raw,name_channel,sf,path_KC)
    noKCs = get_noKCs(raw,name_channel,sf, sN = s_N)

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC
    y = np.concatenate((y_yesKCs,y_noKCs))
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))
    #X = np.array(X_t).T.tolist()

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    #print('TRAIN SHAPE: x_train=',X_train.shape, ' - y_train=', y_train.shape)
    #print('TEST SHAPE: x_test=', X_test.shape, ' - y_test=', y_test.shape)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    #clf.fit(X_train, y_train)
    #print('SCORE:', clf.score(X_test, y_test))

    #svc_disp = plot_roc_curve(clf, X, y)
    #plt.show()

    
    #scores = cross_val_score(clf, X, y, cv=10)
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    print('Scores with 5-CV: ', scores)

    #svc_disp = plot_roc_curve(clf, X, y)
    #plt.show()

    return clf

def KCs(raw,name_channel, sf, filetxt): #Return a vector with only KCs labeled (everything else is zero)
    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])
    filt = filtering(raw, name_channel,sf)
    dat = filt[0]
    num_KCs = 0

    for x in filetxt:
        if x[-3] == 'K' and x[-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = dat[start:end]
            num_KCs += 1

    print('Number of KCs in this candidate = ', num_KCs)

    return data_out

def pulse(time_shape,sfreq): #To generate 0.5 sec x-axes
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    # Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # Pulse signal
    return pulso

def extract_S2(data, sf, filetxt): #Return a list with all stages with phase 2 labeling (together) 
    p2vector_out =[]
    for x in filetxt:
        if x[len(x)-4:len(x)-1] == '2.0':
            vector, cant_p2 = extract_td_phase2(x)
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            
            for i in range(start,end):
                p2vector_out.append(data[i])
                
    return p2vector_out

def plot_KC(raw, name_channel, numKC, path_KC):

    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])
    timeseries = getKCs(raw,name_channel, sf, path_KC)
    data_out[0,200:int(200+2*sf)] = timeseries[numKC]

    return data_out

def re_esctructure(raw, path_KC): #Re-estructure data
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]
    filetxt = open(path_KC, "r")  # Read txt file
    
    pos_c3 = (raw.ch_names).index('C3_1')
    c3_1 = data[pos_c3,:]
    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    step=int(sfreq/4) #every half second

    new_data=data.copy()
    new_data[0]= pulse(time_shape,sfreq)
    new_data[1]= plot_KC(raw, 'C4_1', 2, path_KC) 
    new_data[2]= plot_KC(raw, 'C4_1', 2, path_KC)
    new_data[3]=  plot_KC(raw, 'C4_1', 2, path_KC)
    new_data[4] = plot_KC(raw, 'C4_1', 2, path_KC)
    new_data[5] =  pulse(time_shape,sfreq)
    
    filetxt.close()

    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['Pulse', 'EEG', 'FiltJA', 'Old', 'Tononi', 'FfiltJA']
    new_chtypes = ['misc'] + 4*['eeg'] +['misc'] # Remake channels
    
    # Initialize an info structure      
    new_info = mne.create_info(new_ch_names, sfreq=sfreq, ch_types=new_chtypes)
    new_info['meas_date'] = raw.info['meas_date']      # Record timestamp for annotations
    
    new_raw=mne.io.RawArray(new_data, new_info)        # Build a new raw object 
    new_raw.set_annotations(raw.annotations)         
    
    return new_raw

def plot(raw,n_channels,scal,order): 
    """To visualize the data"""
    raw.plot(show_options=True,
    title='Etiquetado',
    start=0,                        # Initial time to show
    duration=30,                    # Time window (sec) to plot in a given time
    n_channels=n_channels, 
    scalings=scal,                  # Scaling factor for traces.
    block=True,
    show_scalebars = True,
    order=order)

############# Main function ##########################################################
def main():  # Wrapper function
    messagebox.showinfo(message ="This program allows you to tag a specific event.", title="Info")

    path = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS40.vhdr' #easygui.fileopenbox(title ='Select VHDR file.')
    raw= load_brainvision_vhdr(path) 
    show_info(raw)
    path_states = 'C:/Users/aylin/PFC/Señales/TXT/ExpS40.txt' #easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') # To select txt file of previous anotations
    raw,_ = set_sleep_stages(raw,path_states)
    
    path_KC = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS40_2020September29_11-10.txt'#easygui.fileopenbox(title ='Select .txt file with KCs positions and durations.')
    # ExpS11_2020September28_19-48.txt
    # ExpS12_2020September28_20-01.txt
    # ExpS17_2020September28_20-14.txt
    # ExpS21_2020September28_20-29.txt
    # ExpS28_2020September28_20-40.txt
    # ExpS36_2020September29_14-00.txt
    # ExpS37_2020September29_11-01.txt
    # ExpS40_2020September29_11-10.txt
       
    raw_end = re_esctructure(raw, path_KC)
    
    #show_info(raw_end)
    
    #For actual EEG/EOG/EMG/STIM data different scaling factors should be used.
    scal = dict(eeg=1, misc=10e-5) #, stim=10e-5)
    n_channels = 6
    order = [0,1,2,3,4,5]

    print('-------------------------------------------------------------------')
    print()
    clasification(raw,'C4_1',200,path_KC, s_N = 40)
    print()
    print('-------------------------------------------------------------------')

    # 12 - 17 - 21 - 36 - 40
    # Plot it!
    #plot(raw_end,n_channels,scal,order)

    # 50% training - 50% test
        # S12 --> Scores with 5-CV:  [0.71428571 0.57142857 0.92857143 0.85714286 0.85714286]
        # S17 --> Scores with 5-CV:  [0.9375 0.875  1.     0.875  0.875 ]
        # S21 --> Scores with 5-CV: [0.85 0.85 0.6  0.9  1.  ]
        # S36 --> Scores with 5-CV:  [0.92307692 0.96153846 1.         0.92307692 0.88461538]
        # S40 --> Scores with 5-CV: [0.58333333 0.66666667 0.66666667 0.5        0.75      ]


    # 70% training - 30% test
        # S12 --> Scores with 5-CV:  [0.77777778 0.88888889 0.88888889 1.         0.77777778]
        # S17 --> Scores with 5-CV:  [1.  1.  0.9 0.8 0.8]
        # S21 --> Scores with 5-CV: [0.75       0.91666667 0.66666667 0.83333333 1.        ]
        # S36 --> Scores with 5-CV:  [0.9375 0.9375 1.     1.     0.9375]
        # S40 --> Scores with 5-CV: [1.    0.875 0.75  0.5   0.875]


if __name__ == '__main__':
    main()