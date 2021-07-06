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
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import ShuffleSplit
from random import shuffle


#Print the system information
mne.sys_info()

def file_name(path):
    name=os.path.splitext(os.path.basename(path))[0]
    if '_2020' in name:
        ind = str.index(name,'_2020')
        name=name[:ind]

    name=name + datetime.now().strftime("_%Y%B%d_%H-%M") 
    return name

def load_brainvision_vhdr_BBDD(path):
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
    
    #Give Channels
    print('Channels:',raw.info['ch_names'])
    print('--------------> RAW INFO ', raw.info['ch_names'])
    print('EOG: ', raw.copy().pick_types(eog=True).info['ch_names'])
    #Brainvision EMG son misc pero lo cambie a emg
    print('EMG: ', raw.copy().pick_types(emg=True).info['ch_names'])     
    print('Time min: %s seg. Time max: %s seg. ' % (raw.times.min(), raw.times.max()))
    print()

def extract_data(raw,name_channel): #Extracting data, sampling frequency and number of samples
    data, sf, chan =raw._data, raw.info['sfreq'], raw.info['ch_names']   
    n = data.shape[1]   # Number of samples  
    channel = (raw.ch_names).index(name_channel)
    data=data[channel][:]    
    data=data.tolist() # Data (type = list)
    
    return data, sf, n

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

def pulse(time_shape,sfreq):
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    #Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # pulse signal
    return pulso

def LPF(senal, sf, low_pass = 'pad'):  #Low pass filter (Butterworth - Fc = 30 Hz - Order = 5)
    order = 5
    sampling_freq = sf
    cutoff_freq = 30
    sampling_duration = len(senal)
    number_of_samples = sampling_freq * sampling_duration
    time = np.linspace(0, sampling_duration, int(number_of_samples), endpoint=False)

    normalized_cutoff_freq = 2*30/sampling_freq #Normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    order_filter=5

    b, a  = signal.butter(order_filter, normalized_cutoff_freq, output ='ba')  #Return two arrays: numerator and denominator coefficients of the filter.

   
    filtered_signal = signal.filtfilt(b, a, senal, method = low_pass)
    
    return filtered_signal

def BP(senal, sf, notpad = True, band_pass = 'constant'): #Band pass filter (Chebyshev II - Fc1 = 0.125 Hz - Fc2 = 6 Hz - Order = 16)
    order = 16
    rs= 40   #The minimum attenuation required in the stop band. Specified in decibels, as a positive number.
    Wn= [0.125, 6]   #Critical frequencies (For type II filters, this is the point in the transition band at which the gain first reaches -rs. 
    #For digital filters, Wn are in the same units as fs. Wn is in half-cycles / sample.

    sos = signal.cheby2(order, rs, Wn, btype='bandpass', analog=False, output='sos', fs=sf)
    
    if notpad:
        filtered = signal.sosfiltfilt(sos, senal) # default = no pad
    else:
        filtered = signal.sosfiltfilt(sos, senal, padtype = band_pass)

    return filtered

def filtering(raw,name_channel = 'C4_1', sf = 200, overlapping = 50, win = 5): #Filering with windows of five seconds and selected overlapping

    # Overlapping = 1 to 99 --> average of previous filtering and the new one

    data, fs, n = extract_data(raw, name_channel)
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
            data_LP = LPF(data_downsampled, sf)
            # 3° Band-pass Cheby II 0.5-4 Hz
            x = BP(data_LP, sf)
            # 4° Upsampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne 
            dat = signal.resample(x, len(vec), t=None, axis=0, window=None) 
            
            data_out[0,0:int(win*sf)] = dat

        else:

            if pos + sf * win < n-1:  #Each 5 second window instead of the last one
           
                vec = data_referenced[pos:pos+int(win*sf)]

                # 1° Downsampling 200 Hz --> 100 Hz
                data_downsampled = signal.decimate(vec, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
                # 2° Low-pass Fc = 30 Hz
                data_LP = LPF(data_downsampled, sf)
                # 3° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, sf)
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
                data_LP = LPF(data_downsampled, sf)
                # 4° Band-pass Cheby II 0.5-4 Hz
                x = BP(data_LP, sf)
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

def KCs(filtered_signal,name_channel, path_KC, sf = 200): #Return a vector with only KCs labeled (everything else is zero)

    data = filtered_signal[0]
    data = data.tolist()
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
    KC = data[start:end].tolist()
    c = center(KC)
    cent = c + start
    mid_dur = (2 - (start - end)) // 2
    starting = cent - int(sf*d/2)
    ending = starting + d * sf

    a_KC = data[int(starting):int(ending)]

    return a_KC, len(a_KC)

def align_KC_list(start, end, data, sf = 200):
    d = 2 #in seconds
    dur = d * sf
    KC = data[0,start:end]
    KC_list = KC.tolist()
    c = center(KC_list)
    cent = c + start
    mid_dur = (2 - (start - end)) // 2
    starting = cent - int(sf*d/2)
    ending = starting + d * sf

    a_KC = data[0,int(starting):int(ending)]
    #a_KC = signal.resample(a_KC, , t=None, axis=0, window=None) 

    return a_KC, len(a_KC)

def getKCs_list(data, name_channel = 'C4_1', sf = 200, pathKC = ''): 
    
    timeseries = []

    filetxt = open(pathKC, 'r')

    for x in filetxt:
        if x[-3] == 'K' and x[-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            a_KC, leng = align_KC_list(start, end, data, sf)
            timeseries.append(a_KC.tolist())
    
    filetxt.close()
    return timeseries

def getKCs(filt, name_channel = 'C4_1', sf = 200, pathKC = ''): #Return a vector with only KCs labeled (everything else is zero)
    
    timeseries = []

    for j in range(0, len(filt)):
        data = filt[j][0]
        path_KC = pathKC[j]
        filetxt = open(path_KC, 'r')

        for x in filetxt:
            if x[-3] == 'K' and x[-2] == 'C':
                vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
                start = int(vector[0,0]*sf)
                end = int((vector[0,0] + vector[0,1])*sf)
                a_KC, leng = align_KC(start, end, data, sf)
                timeseries.append(a_KC.tolist())
        
        filetxt.close()
    
    return timeseries

def get_noKCs(filt, name_channel, sf, s_N = [11,12,17,21,28,36,37,40]):
    
    noKCs = []
    
    for sN in s_N:
        if sN == 11:
            pos_inic_noKC = [1124,1180,1227,1259,1283,1379,1520.5,1526.5,1653]
            data = filt[0][0]

        elif sN == 12:
            pos_inic_noKC = [1005,1180,1220,1320,1360,1485,1585,1645,1730,1860,1945,2020,2130,2215]
            #pos_inic_noKC = [1005,1180,1255.5,1288,1300.5,1331.5,1507.5,1529,1655,1743,1770,1789,1801,1900.5]
            data = filt[1][0]

        elif sN == 17:
            pos_inic_noKC = [910,920,995,1050,1140,1175,1235,1275,1330,1375,1445,1525,1565,1660,1730,2033]
            #pos_inic_noKC = [910,970,979.5,988.5,1145,1160.5,1230.5,1251,1320,1336,1436.5,1444,1466,1504,1506,1534]
            data = filt[2][0]

        elif sN == 21:
            pos_inic_noKC = [575,605,690,740,805,845,920,955,1015,1090,1160,1200,1255,1260,2715,2770,2805,2860,2875,2905]
            #pos_inic_noKC = [635.5,663.5,854.5,740,878.5,962,980,1006,1052.5,1055.5,1085.5,1096.5,1148.5,1200,1204.5,1213,1219,1234,1240.5,1258.5]
            data = filt[3][0]

        elif sN == 28:
            pos_inic_noKC = [293,318,378.5,505,582.5,666.5,793.5]
            data = filt[4][0]

        elif sN == 36:
            pos_inic_noKC = [435,500,620,750,820.5,1065,1180,1270,1340,1510,1660,1825,1900,1980,2018,2100,2130,2180,2420,2600,3035,3350,3540,3665,3875,4100]
            #pos_inic_noKC = [542,784.5,804,1083,1117.5,1129,1182,1211,1273.5,1295.5,1305.5,1324.5,1344.5,1434,1493.5,1509,1589,1636.5,1664,1740.5,1798,1817,1927,2203,2221,2309]
            data = filt[5][0]
        
        elif sN == 37:
            pos_inic_noKC = [1776.5,2055]
            data = filt[6][0]

        elif sN == 40:
            pos_inic_noKC = [755,925,945,1040,1090,1160,1200,1255,1295,1370,1495,1840]
            #pos_inic_noKC = [767,774,921,960,1021,1044,1160.5,1187,1329,1364,1744,1844]
            data = filt[7][0]
        else:
            print('Do not use S',sN, ' because there are no KCs enough to make a pipeline')

        for start in pos_inic_noKC:
            noKCs.append(data[int(start):int(start+2*sf)]) #tolist()

    return noKCs

def clasification(x , list_testing, name_channel = 'C4_1',sf = 200, path_KCbbdd = ''):
    
    yesKCs = getKCs(x, name_channel = 'C4_1', sf = sf, pathKC = path_KCbbdd)
    noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC

    y = np.concatenate((y_yesKCs,y_noKCs))
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
    scores0 = cross_val_score(clf, X, y, cv=cv)
    print('Puntajes del modelo a con una validación cruzada de 10 splits y tomando a la mitad de los datos como entrenamiento:\n ', scores0)
    clf.fit(X,y)
    return clf

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

def put_flag(filt, no_filt, sf = 200, name_channel = 'C4_1', path_KC = ''):
    
    x = filt[0]
    x_nofilt = no_filt
    tamano = x.shape[0]
    flags = np.zeros(tamano)
    window = 2
    if_S2 = detect_if_S2(path_KC)
    cant = 0
    
    for start in if_S2:
        inicio = int(start*sf)
        final = inicio + int(30*sf)
        i = inicio

        while i < final:
            data = x_nofilt[i:int(i+window*sf)].tolist()
            maxi = max(data)
            mini = min(data)
            pos_maxi = data.index(maxi)
            pos_mini = data.index(mini)
            maxi = maxi * (10**6)
            mini = mini * (10**6)
            a_pp = (maxi - mini)
            t_mini_maxi = (pos_maxi - pos_mini)/sf

            if (pos_mini<pos_maxi) and (a_pp>75) and (maxi>20) and (mini<-30) and (t_mini_maxi<1) and (t_mini_maxi>0.1):
                c = (pos_maxi + pos_mini) // 2
                flags[i+int(c-sf):i+int(c+sf)] = 25e-6
                i = i + int(window*sf*0.25)
                cant = cant + 1
            else:
                i = i + int(window*sf*0.25)
        
    return flags, cant

def plot_KC_aligned(filt, name_channel, numKC, path_KC, sf = 200):

    data_out = np.zeros([1,filt.shape[1]])
    timeseries = getKCs_list(filt,name_channel, sf, path_KC)
    data_out[0,sf:int(sf+2*sf)] = timeseries[numKC]

    return data_out

def separate_candidates(pos_candidates):
    inicio = 0
    pos_candidates = pos_candidates[0]
    candidates_separated = []
    ult = len(pos_candidates)
  
    for i in range(0, ult-1):
        this_value_pos = pos_candidates[i]
        next_value_pos = pos_candidates[i+1]

        if next_value_pos == pos_candidates[-1]:
            candidates_separated.append(pos_candidates[inicio:])

        if this_value_pos != next_value_pos - 1:
            candidates_separated.append(pos_candidates[inicio:i+1])
            inicio = i+1    

    return candidates_separated #list

def detector(clf, filtered_signal, flags, sf=200):

    data_out =np.zeros([1,filtered_signal.shape[1]])
    pos_cand = np.where(flags)
    candi = separate_candidates(pos_cand)
    cant_KC = 0

    for a in candi:
        if len(a) == 2*sf:
            candidate0 = filtered_signal[0,int(a[0]):int(a[0]+2*sf)]
            cand = np.array(candidate0)
            candi = signal.resample(cand, 400)
            candidate = np.reshape(candi, (1,400))
            p = clf.predict(candidate)

            if p == 1: #yes KC
                cant_KC = cant_KC + 1
                data_out[0,int(a[0]):int(a[0]+2*sf)] = 25e-6

        elif len(a) > 2*sf:
            i = a[0]
            while i < a[-1]:
                candidate0 = filtered_signal[0,int(i):int(i+2*sf)]
                cand = np.array(candidate0)
                candi = signal.resample(cand, 400)
                candidate = np.reshape(candi, (1,400))
                p = clf.predict(candidate)

                if p == 1: #yes KC
                    cant_KC = cant_KC + 1
                    data_out[0,int(i):int(i+2*sf)] = 25e-6
                    i = i+2*sf

                elif p == 0: # no KC
                    i = i + sf/8

    return data_out, cant_KC

##Re-estructure data
def re_esctructure(raw, filtered_signal, detected, path_KC):
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]
    
    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    new_data=data.copy()

    new_data[0]= pulse(time_shape,sfreq)
    new_data[1]= c4_1
    new_data[2] = filtered_signal
    new_data[3]= KCs(filtered_signal, 'C4_1', sf = 200, path_KC=path_KC)
    new_data[4], _ = put_flag(filtered_signal, c4_1, path_KC = path_KC)
    new_data[5]= detected

    new_data=new_data[[0,1,2,3,4,5], :]


    new_ch_names = ['Pulse', 'C4', 'Filtrada', 'KC labeled', 'Flags', 'KC detected'] #, 'Filtrada','Supera 75']  

    new_chtypes = ['misc'] + 5*['eeg']
    
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
    start=0,                        # initial time to show
    duration=30,                    # time window (sec) to plot in a given time
    n_channels=n_channels, 
    scalings=scal,                  # scaling factor for traces.
    block=True,
    order=order)

#Main function
def main():  # Wrapper function
    #messagebox.showinfo(message="This program allows you to tag a specific event.", title="Info")

    path = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS11.vhdr'
    #'C:/Users/aylin/PFC/Experimentacion_nuestra/Señales/Antes/ancarosi1.vhdr'#easygui.fileopenbox(title='Select VHDR file.')#selecciono la carpeta vhdr
    raw=load_brainvision_vhdr_BBDD(path) 

    path_states = 'C:/Users/aylin/PFC/Señales/TXT/ExpS11.txt'
    #'C:/Users/aylin/PFC/Experimentacion_nuestra/Scoring/ancarosi1.txt'#easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') #selecciono el txt de anotaciones anteriores
    raw_0,_ = set_sleep_stages(raw,path_states)
   
    path_KC1 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS11_2020September28_19-48.txt' 
    #'C:/Users/aylin/PFC/Experimentacion_nuestra/KC ETIQUETADOS/ancarosi1_2021March09_18-48.txt' #easygui.fileopenbox(title='Select the labeling file (with extension txt).')
    
    filtered_signal = filtering(raw_0, sf = 200, name_channel='C4_1')

    list_test = getKCs_list(filtered_signal, name_channel='C4_1', sf = 200, pathKC = path_KC1)
    list_testing = np.array(list_test)
    #list_testing = signal.resample(list_testing, 400, axis=1) #n_samples , n_features

   
   ########################################################################################################################
    ## S11
    print('Cargando datos del Sujeto 11')
    path11 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS11.vhdr'
    raw11 = load_brainvision_vhdr_BBDD(path11) 
    path_KC11 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS11_2020September28_19-48.txt'       
    data11,sfreq11 =raw11.get_data(),raw11.info['sfreq']

    ## S12
    print('Cargando datos del Sujeto 12')
    path12 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS12.vhdr'
    raw12 = load_brainvision_vhdr_BBDD(path12) 
    path_KC12 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS12_2020September28_20-01.txt'     
    data12,sfreq12 =raw12.get_data(),raw12.info['sfreq']

    ## S17
    print('Cargando datos del Sujeto 17')
    path17 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS17.vhdr'
    raw17 = load_brainvision_vhdr_BBDD(path17) 
    path_KC17 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS17_2020September28_20-14.txt'       
    data17,sfreq17 =raw17.get_data(),raw17.info['sfreq']

    ## S21
    print('Cargando datos del Sujeto 21')
    path21 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS21.vhdr'
    raw21 = load_brainvision_vhdr_BBDD(path21) 
    path_KC21 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS21_2020September28_20-29.txt'       
    data21,sfreq21 =raw21.get_data(),raw21.info['sfreq']

    ## S28
    print('Cargando datos del Sujeto 28')
    path28 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS28.vhdr'
    raw28 = load_brainvision_vhdr_BBDD(path28) 
    path_states28 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS28.txt'
    path_KC28 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS28_2020September28_20-40.txt'       
    data28,sfreq28 =raw28.get_data(),raw28.info['sfreq']

    ## S36
    print('Cargando datos del Sujeto 36')
    path36 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS36.vhdr'
    raw36 = load_brainvision_vhdr_BBDD(path36) 
    path_states36 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS36.txt'
    path_KC36 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS36_2020September29_14-00.txt'       
    data36,sfreq36 =raw36.get_data(),raw36.info['sfreq']

    ## S37
    print('Cargando datos del Sujeto 37')
    path37 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS37.vhdr'
    raw37 = load_brainvision_vhdr_BBDD(path37) 
    path_states37 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS37.txt'
    path_KC37 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS37_2020September29_11-01.txt'       
    data37,sfreq37 =raw37.get_data(),raw37.info['sfreq']    

    ## S40
    print('Cargando datos del Sujeto 40')
    path40 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS40.vhdr'
    raw40 = load_brainvision_vhdr_BBDD(path40) 
    path_states40 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS40.txt'
    path_KC40 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS40_2020September29_11-10.txt'       
    data40,sfreq40 =raw40.get_data(),raw40.info['sfreq']

    print('------------------------------------------------------------------------')

    data = [data11, data12, data17, data21, data28, data36, data37, data40]
    raw = [raw11, raw12, raw17, raw21, raw28, raw36, raw37, raw40]
    path_KC = [path_KC11,path_KC12,path_KC17,path_KC21,path_KC28,path_KC36,path_KC37,path_KC40]
    filt = []

    print('Filtrando todas las señales de la BBDD:')

    for num in range(0,len(raw)):
        r = raw[num]
        f = filtering(raw = r, name_channel = 'C4_1', sf = 200)
        filt.append(f)
        print('...........................................................................................')
    
    print('Finalizó el filtrado de todas las señales de la BBDD \n')

    print('Comenzando a entrenar el modelo con los datos de la BBDD')
    clf = clasification(filt, list_testing , path_KCbbdd = path_KC)
    print()
    print('Localizando los posibles candidatos de KCs')
    no_filtered, sfreq =raw_0.get_data(),raw_0.info['sfreq']    
    pos_c4 =(raw_0.ch_names).index('C4_1')
    c4_1 = no_filtered[pos_c4,:]
    candidates, cant_flags = put_flag(filtered_signal, c4_1, path_KC = path_KC1)
    print('Se detectaron ', cant_flags, ' posibles candidatos a KC')
    print()
    print('Predeciendo cuáles son KC y cuáles no')
    detected, cant_KC = detector(clf,filtered_signal,candidates, sf = 200)    
    print('Se detectaron ', cant_KC, ' KCs \n')

    # Plot it!
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3, stim=2)
    n_channels = 6
    order=[0,1,2,3,4,5]
    raw_signal = re_esctructure(raw_0, filtered_signal, detected, path_KC1)
    plot(raw_signal,n_channels,scal,order)

    print('------------------------------------------------------------------------')

    print('Goodbye =)')

if __name__ == '__main__':
    main()