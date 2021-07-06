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

    if low_pass == 'pad':
        filtered_signal = signal.filtfilt(b, a, senal, method = low_pass)
    else:
        filtered_signal = signal.filtfilt(b, a, senal)
    
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

def filtering_nowindow(raw,name_channel,sf, bandpass_ = 'constant', lowpass_ = 'pad'): #Filering all signal --> Tononi
    data, sf, n = extract_data(raw, name_channel)  
    data_out = np.zeros([1,len(data)])

    # 1° Downsampling 200 Hz --> 100 Hz
    data_downsampled = signal.decimate(data, 2, n=None, ftype='iir', axis=- 1, zero_phase=True)
    # 2° Signal - avg(signal)
    data_referenced =  data_downsampled - np.mean(data_downsampled)
    # 3° Low-pass Fc = 30 Hz
    data_LP = LPF(data_referenced, low_pass='otro')
    # 4° Band-pass Cheby II 0.5-4 Hz
    x = BP(data_LP, notpad=True)
    # 5° Oversampling 100 Hz --> 200 Hz   //// To plot this signal, otherwise it can't be plotted with mne tools
    dat = signal.resample(x, len(data), t=None, axis=0, window=None) 

    data_out[0,:] = dat   

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

def filteredKCs(raw,name_channel, sf, filetxt): #Return a vector with only KCs labeled (everything else is zero)
    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])
    x = filtering(raw, name_channel,sf)
    data = x[0]

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = data[start:end]

    return data_out

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

def SNR_algorithm_filt(signal, filtered, sf, path_KC):
    filetxt = open(path_KC, "r")
    SNRs = []
    for line in filetxt:
        if line[len(line)-3] == 'K' and line[len(line)-2] == 'C':
            vector, cantKC = extract_td(line)  
            start = int(vector[0,0]* sf)
            end = int(start + vector[0,1]*sf)
            Kcomp = signal[start:end] # noise + signal
            Kcomp = np.array(Kcomp)    
            kc_signal = filtered[start:end] # signal
            kc_signal = abs(np.array(kc_signal))
            kc_noise = abs(Kcomp - kc_signal)
            kc_SNR = kc_signal/kc_noise
            kc_SNR = kc_SNR.tolist()
            SNRs.extend(kc_SNR)

    SNR = np.mean(np.array(SNRs))    

    return SNR

def SNR_filtJuliAylin_new(raw, name_channel, sf, path_KC, overlap, bp, lp): #Calculate SNR with the signal filt
    filtered = filtering(raw,name_channel,sf, overlapping = overlap, bandpass_= bp, lowpass_ = lp)
    filtered = filtered.tolist()
    filtered = filtered[0]
    data, sf, n = extract_data(raw, name_channel)
    SNRJA = SNR_algorithm_filt(data, filtered, sf, path_KC)

    return SNRJA

def SNR_filtTononi(raw,name_channel, sf, path_KC, bp, lp):
    filtered = filtering_nowindow(raw,name_channel,sf, bandpass_= bp, lowpass_ = lp)
    filtered = filtered.tolist()
    filtered = filtered[0]
    data, sf, n = extract_data(raw, name_channel)
    SNRT = SNR_algorithm_filt(data, filtered, sf, path_KC)

    return SNRT

def SNR_varp2p_algorithm(signal, sf, path_KC):
    filetxt = open(path_KC, "r")
    SNRs = []
    for line in filetxt:
        if line[len(line)-3] == 'K' and line[len(line)-2] == 'C':
            vector, cantKC = extract_td(line)  
            start = int(vector[0,0]* sf)
            end = int(start + vector[0,1]*sf)
            Kcomp = signal[start:end] # noise + signal
            # Calculating Peak 2 Peak 
            p2p = max(Kcomp) - min(Kcomp)
            # Calculating variance
            vari = np.var(np.array(Kcomp))
            SNRs.append(p2p/vari)

    SNR = np.mean(np.array(SNRs)) 

    return SNR

def SNR_varp2p_signal(raw, name_channel, sf, path_KC):
    data, sf, n = extract_data(raw, name_channel)
    SNR =  SNR_varp2p_algorithm(data, sf, path_KC)
    return SNR

def SNR_varp2p_JuliAylin_new(raw, name_channel, sf, path_KC, overlap, bp, lp):
    # SNR_varp2p_JuliAylin_new(raw, 'C4_1', 200, path_KC, overlap, bp, lp)
    filtered = filtering(raw,name_channel, sf, overlapping = overlap, bandpass_= bp, lowpass_ = lp)
    filtered = filtered.tolist()
    filtered = filtered[0]
    SNR = SNR_varp2p_algorithm(filtered, sf, path_KC)
    return SNR

def SNR_varp2p_Tononi(raw, name_channel, sf, path_KC, bp, lp):
    filtered = filtering_nowindow(raw,name_channel,sf, bandpass_= bp, lowpass_ = lp)
    filtered = filtered.tolist()
    filtered = filtered[0]
    SNR = SNR_varp2p_algorithm(filtered, sf, path_KC)
    return SNR

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
    new_data[1]= c4_1 
    new_data[2]= filtering(raw, 'C4_1', 200)
    new_data[3]= filtering_nowindow(raw, 'C4_1', 200)
    new_data[4]= KCs(raw, 'C4_1', 200, path_KC)
    new_data[5]=  pulse(time_shape,sfreq)
    
    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['Pulse-1', 'C4', 'C-V', 'R', 'KC', 'Pulse-2']
    new_chtypes = ['misc'] + 4*['eeg'] + ['misc'] # Remake channels
    
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
    order=order)

############# Main function ##########################################################
def main():  # Wrapper function
    messagebox.showinfo(message ="This program allows you to calculate different kinds of SNR.", title="Info")

    path = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS21.vhdr' #easygui.fileopenbox(title ='Select VHDR file.')
    raw = load_brainvision_vhdr(path)
    show_info(raw)
    path_states = 'C:/Users/aylin/PFC/Señales/TXT/ExpS21.txt' #easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') # To select txt file of previous anotations
    raw,_ = set_sleep_stages(raw,path_states)
    
    path_KC = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS21_2020September28_20-29.txt' #easygui.fileopenbox(title ='Select .txt file with KCs positions and durations.')
    
    ### ExpS11_2020September28_19-48.txt
    ### ExpS12_2020September28_20-01.txt
    ### ExpS17_2020September28_20-14.txt
    ### ExpS21_2020September28_20-29.txt
    ### ExpS28_2020September28_20-40.txt
    ### ExpS36_2020September29_14-00.txt
    ### ExpS37_2020September29_11-01.txt
    ### ExpS40_2020September29_11-10.txt
    raw_end = re_esctructure(raw, path_KC)
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3, stim=2)
    n_channels = 6
    order=[0,1,2,3,4,5]
    plot(raw_end,n_channels,scal,order)
    
    """bp = 'constant'
    lp = 'gust'

    ov = 25
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 25: ', stop - start)  

    ov = 50
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 50: ', stop - start)

    ov = 75
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 75: ', stop - start) 


    #raw.annotations.save(file_name(path)+ ".txt")
    namefile = 'S11.xls'

    bp = 'constant'
    lp = 'gust'

    ov = 25
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 25: ', stop - start)  

    ov = 50
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 50: ', stop - start)

    ov = 75
    start = timeit.default_timer()
    SNRJA = filtering(raw, 'C4_1', 200, overlapping = ov, win = 5, bandpass_ = bp, lowpass_ = lp)
    stop = timeit.default_timer()
    print('Time 75: ', stop - start) 

    workbook = xlsxwriter.Workbook(namefile)
    worksheet = workbook.add_worksheet()
    list_SNR = []

    print('---------------------------------------------------------------')
    print()
    print('Calculating SNR P2P/VAR')
    print()
    
    SNRE = SNR_varp2p_signal(raw,'C4_1', 200, path_KC)
    
    SNRJA_list = []
    SNRT_list = []

    for overlap in [25,50,75]:
        for bp in ['constant', 'odd', 'even']:
            for lp in ['gust', 'pad']:
                SNRJA = SNR_varp2p_JuliAylin_new(raw, 'C4_1', 200, path_KC, overlap, bp, lp)
                SNRJA_list.append(SNRJA)
            
                SNRT = SNR_varp2p_Tononi(raw, 'C4_1', 200, path_KC, bp, lp)
                SNRT_list.append(SNRT)
    print('----------------------------------------------------------------')
    
    print()
    print('Calculating SNR FILTRADA/(RAW-FILTRADA)')
    
    SNRJA_f_list = []
    SNRT_f_list = []

    for overlap in [25,50,75]:
        for bp in ['constant', 'odd', 'even']:
            for lp in ['gust', 'pad']:

                SNRJA_f = SNR_filtJuliAylin_new(raw, 'C4_1', 200, path_KC, overlap, bp, lp)
                SNRJA_f_list.append(SNRJA_f)
                
                SNRT_f = SNR_filtTononi(raw, 'C4_1', 200, path_KC, bp, lp)
                SNRT_f_list.append(SNRT_f)

    print('----------------------------------------------------------------')

    print('Extracting data to a .xlsx file called: ', namefile)

#############################################################################
######    CREATING XLSX FILE  ###############################################
#############################################################################  
    worksheet.write(0, 0, '(Var/P2P) Juli-Aylin')
    #col = 1
    #for titulo in list_title:
    #    worksheet.write(0, col, titulo)
    #    col += 1
    row = 0
    col = 1
    for SNR in SNRJA_list:
        worksheet.write(row, col, SNR)
        col += 1    
#############################################################################  
    worksheet.write(3, 0, '(Var/P2P) Tononi')
    #col = 1
    #for titulo in list_title:
    #    worksheet.write(3, col, titulo)
    #    col =+ 1   
    row = 3
    col = 1
    for SNR in SNRT_list:
        worksheet.write(row, col, SNR)
        col += 1     
#############################################################################
    worksheet.write(6, 0, '(S/R) Juli-Aylin')
    #col = 1
    #for titulo in list_title_f:
    #    worksheet.write(6, col, titulo)
    #    col =+1    
    row = 6
    col = 1
    for SNR in SNRJA_f_list:
        worksheet.write(row, col, SNR)
        col += 1         
#############################################################################    
    worksheet.write(9, 0, '(S/R) Tononi')
    #col = 1
    #for titulo in list_title_f:
    #    worksheet.write(9, col, titulo)
    #    col =+1
    row = 9
    col = 1
    for SNR in SNRT_f_list:
        worksheet.write(row, col, SNR)
        col += 1 
#############################################################################
    worksheet.write(12, 0, '(Var/P2P) EEG')
    worksheet.write(12, 1, SNRE)


    workbook.close()
#############################################################################
#############################################################################
#############################################################################  

    # Plot it!"""

if __name__ == '__main__':
    main()