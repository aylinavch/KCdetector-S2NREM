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
            #eog=('EOG1_1','EOG2_1'),
            #misc=('EMG1_1','EMG2_1'),
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
    #print('Channels:',raw.info['ch_names'])
    #print('EEG: ', list(raw.copy().pick_types(eeg=True).info['ch_names']))
    #print('EOG: ', raw.copy().pick_types(eog=True).info['ch_names'])

    #Brainvision EMG son misc pero lo cambie a emg
    #print('EMG: ', raw.copy().pick_types(emg=True).info['ch_names'])     
    #print('Time min: %s seg. Time max: %s seg. ' % (raw.times.min(), raw.times.max()))
    #print()

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

def filtering(raw, name_channel,sf, overlapping = 50, win = 5, bandpass_ = 'even', lowpass_ = 'gust'): #Filering with windows of five seconds and selected overlapping 
     
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

def KCs(raw,name_channel, sf, filetxt): #Return a vector with only KCs labeled (everything else is zero)
    data, sf, n = extract_data(raw, name_channel)
    data_out = np.zeros([1,len(data)])

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            data_out[0,start:end] = data[start:end]

    return data_out

def pulse(time_shape,sfreq): #To generate 0.5 sec x-axes
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    # Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # Pulse signal

    return pulso

def subtraction_eog(raw): #Substraction of EOGs signals
    eog1= raw.get_data(picks='EOG1_1') 
    eog2= raw.get_data(picks='EOG2_1')  
    sub_eog = eog1-eog2

    return sub_eog

def subtraction_emg(raw): #Substraction of EMGs signals
    emg1= raw.get_data(picks='EMG1_1') 
    emg2= raw.get_data(picks='EMG2_1')   
    sub_emg = emg1-emg2

    return sub_emg

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

def slopes(signal, sf):
    signal = signal.tolist()
    maxi_abs = max(signal)
    mini = min(signal)

    pos_maxi_abs = signal.index(maxi_abs)

    pos_mini = signal.index(mini)

    maxi_loc = max(signal[0:pos_mini])
    pos_maxi_loc = signal.index(maxi_loc)

    slope1 = (maxi_loc - mini)*(10**6)/((pos_mini - pos_maxi_loc)*(10**3)/sf)

    slope2 = (maxi_abs - mini)*(10**6)/((pos_maxi_abs - pos_mini))*(10**3)/sf

    slope3 = (maxi_abs - signal[-1])*(10**6)/((len(signal) - pos_maxi_abs)*(10**3)/sf)

    durhmin = (pos_mini/sf)*(10**3)
    
    durhmaxsup = (pos_maxi_abs/sf)*(10**3)
    
    durhmaxloc = (pos_maxi_loc/sf)*(10**3)

    return slope1, slope2, slope3, durhmin, durhmaxsup, durhmaxloc

def statistics_KCs(raw, pathfile, filenamepath):
    
    filetxt = open(pathfile, "r")  # Read txt file
        
    data_raw, sf, n = extract_data(raw, 'C4_1')
    data = filtering(raw, 'C4_1', sf)[0]

    title_list = ['Slope1', 'Slope2', 'Slope3', 'App', 'Dur total', 'Max', 'Min','Dur h min', 'Dur h max sup', 'Dur h max loc']
    numKC = 0

    slope1_list = []
    slope2_list = []
    slope3_list = []
    app_list = []
    durtotal_list = []
    max_list = []
    min_list = []
    durhmin_list = []
    durhmaxsup_list = []
    durhmaxloc_list = []

    for x in filetxt:
        if x[len(x)-3] == 'K' and x[len(x)-2] == 'C':
            numKC +=1
            vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
            start = int(vector[0,0]*sf)
            end = int((vector[0,0] + vector[0,1])*sf)
            complexK = data[start:end]  # type = list 
            
            # Calculating max and min of each KC
            maxi = max(complexK)
            mini = min(complexK)
            max_list.append(maxi*(10**6))
            min_list.append(mini*(10**6))

            # Calculating App
            p2p = maxi - mini
            app_list.append(p2p)            
            
            # Calculating total duration of each KC
            durtotal_list.append(vector[0,1]*(10**3))

            # Calculating the other features
            slp1, slp2, slp3, durhmin, durhmaxsup, durhmaxloc = slopes(complexK, sf) 
            slope1_list.append(slp1)
            slope2_list.append(slp2)
            slope3_list.append(slp3)
            durhmin_list.append(durhmin)
            durhmaxsup_list.append(durhmaxsup)
            durhmaxloc_list.append(durhmaxloc)
            
    filetxt.close()

    namefile = filenamepath + '.xls'
    workbook = xlsxwriter.Workbook(namefile)
    worksheet = workbook.add_worksheet()
    
    worksheet.write(0, 0, 'Cant KCs')
    worksheet.write(0, 1, numKC)

    row = 2
    for i in range(1,numKC+1):
        name = 'KC' + str(i)
        worksheet.write(row, 0, name)
        row = row + 1

    col = 1
    for title in title_list:
        worksheet.write(1, col, title)
        col = col + 1  

    row = 2
    for s1 in slope1_list:
        worksheet.write(row, 1, s1)
        row = row + 1

    row = 2
    for s2 in slope2_list:
        worksheet.write(row, 2, s2)
        row = row + 1
    
    row = 2
    for s3 in slope3_list:
        worksheet.write(row, 3, s3)
        row = row + 1

    row = 2
    for ap2p in app_list:
        worksheet.write(row, 4, ap2p)
        row = row + 1

    row = 2
    for dt in durtotal_list:
        worksheet.write(row, 5, dt)
        row = row + 1

    row = 2
    for maximo in max_list:
        worksheet.write(row, 6, maximo)
        row =row + 1

    row = 2
    for minimo in min_list:
        worksheet.write(row, 7, minimo)
        row = row + 1

    row = 2
    for dmin in durhmin_list:
        worksheet.write(row, 8, dmin)
        row = row + 1
    
    row = 2
    for durhms in durhmaxsup_list:
        worksheet.write(row, 9, durhms)
        row = row + 1
    
    row = 2
    for durhml in durhmaxloc_list:
        worksheet.write(row, 10, durhml)
        row = row + 1

    workbook.close()

def extract_td_stages(x):
  l_string = x.split(',')
  l = []
  for i in l_string:
     l.append(float(i))
  return l  # l = [start,duration,stage]

def time_stages(path_KC): #Return a vector with only KCs labeled (everything else is zero)
    filetxt = open(path_KC, 'r')

    s0 = 0    
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0

    for x in filetxt:
        if x[len(x)-3] != 'K' and x[len(x)-2] != 'C' and x[0] != '#':
            lista = extract_td_stages(x[0:-1])  # l = [start,duration,stage]
            stage = lista[-1]

            if stage == 0.0:
                s0 = s0 + 30
            elif stage == 1.0:
                s1 = s1 + 30
            elif stage == 2.0:
                s2 = s2 + 30
            elif stage == 3.0:
                s3 = s3 + 30
            elif stage == 4.0:
                s4 = s4 + 30
  
    print('Duración vigilia: ', s0, 'segundos')
    print('Duración de sueño: ', s1+s2+s3+s4, 'segundos')
    print('     S1: ', s1, 'segundos')
    print('     S2: ', s2, 'segundos')
    print('     S3: ', s3, 'segundos')
    print('     S4: ', s4, 'segundos')
    filetxt.close()

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
    new_data[3]=  c4_1
    new_data[4] = c4_1
    new_data[5]=  KCs(raw, 'C4_1', 200, filetxt) 
    
    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['Pulse', 'EEG', 'FiltJA', 'Old', 'Tononi', 'FfiltJA']
    new_chtypes = ['misc'] + 5*['eeg'] #['eog'] + ['emg'] + ['misc'] + 3*['eeg']         # Remake channels
    
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
    messagebox.showinfo(message ="This program allows you to tag a specific event.", title="Info")

    path = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS11.vhdr' #easygui.fileopenbox(title ='Select VHDR file.')
    raw = load_brainvision_vhdr(path) 
    show_info(raw)

    path_states = 'C:/Users/aylin/PFC/Señales/TXT/ExpS11.txt' #easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') # To select txt file of previous anotations
    raw,_ = set_sleep_stages(raw, path_states)
    
    path_KC = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS11_2020September28_19-48.txt' #easygui.fileopenbox(title ='Select .txt file with KCs positions and durations.')
    # ExpS11_2020September28_19-48.txt
    # ExpS12_2020September28_20-01.txt
    # ExpS17_2020September28_20-14.txt
    # ExpS21_2020September28_20-29.txt
    # ExpS28_2020September28_20-40.txt
    # ExpS36_2020September29_14-00.txt
    # ExpS37_2020September29_11-01.txt
    # ExpS40_2020September29_11-10.txt
    
    statistics_KCs(raw, path_KC, 'S11')
    time_stages(path_KC)
    

    #raw_end = re_esctructure(raw, path_KC)  
    #show_info(raw_end)
    
    #For actual EEG/EOG/EMG/STIM data different scaling factors should be used.
    #scal = dict(eeg=20e-5, misc=1e-3) #, stim=10e-5)
    #n_channels = 6
    #order = [0,1,2,3,4,5]

    # Plot it!
    #plot(raw_end,n_channels,scal,order)
   
    
    # WARNING: This script don't include any line to save data, it is just to visualize different steps. 
    # To save data you can include the following lines: 
    #aw.annotations.save(file_name(path)+ ".txt")
    #raw.save(file_name(path)+  ".fif",overwrite=True)  
    #print('Scoring was completed and the data was saved.')

if __name__ == '__main__':
    main()