import numpy as np
import mne
import os
import matplotlib.pyplot as plt                    
import argparse
import easygui

from datetime import datetime    
from tkinter import messagebox
from scipy import signal
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
            eog=('EOG1_1','EOG2_1'),
            misc=('EMG1_1','EMG2_1'),
            verbose=True)
    raw.rename_channels(lambda s: s.strip("."))

    # Specify this as the emg channel (channel type)
    raw.set_channel_types({'EMG1_1': 'emg','EMG2_1': 'emg'}) 
    print('')
    print('Done!')

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
    print('EEG: ', list(raw.copy().pick_types(eeg=True).info['ch_names']))
    print('EOG: ', raw.copy().pick_types(eog=True).info['ch_names'])
    #Brainvision EMG son misc pero lo cambie a emg
    print('EMG: ', raw.copy().pick_types(emg=True).info['ch_names'])     
    print('Time min: %s seg. Time max: %s seg. ' % (raw.times.min(), raw.times.max()))
    print()

def set_sleep_stages(raw,path_stages):
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

##New Channels
def Supera75(raw):
    data,sfreq =(raw.get_data())* 1e6,int(raw.info['sfreq'])  
    #data=data*1e6  # Convert Volts to uV
    n=len(raw) #len(raw) = data.shape[1]

    arraysenial=np.asarray([])

    for i in range(0,n,sfreq):   
        eeg=data[0][i:i+200]    #Every second
        
        signal=eeg - np.mean(eeg) 
        signal=signal+np.min(signal)*-1 
        
        newsignal=signal
        newsignal[signal>75]=100
        newsignal[signal<=75]=0

        arraysenial=np.concatenate( (arraysenial, list(newsignal)), axis=0) 

    return arraysenial

def upper_than(raw,name_channel,threshold,sf,value):  #STEP IS NOT APPLIED YET!!
    ### Extract data, sampling frequency and channels names
    data,sf,chan=raw._data,raw.info['sfreq'], raw.info['ch_names']
    data=data*1e6       # Convert Volts to uV
    n = data.shape[1]   #samples    
    
    channel = (raw.ch_names).index(name_channel)
    print('Channel choose:',raw.ch_names[channel])
    
    step=int(sf/2)
    win=200  ##200 =1 sec 
    zeros=[0 for i in range(0,step)]
    ones=[1 for i in range(0,step)]

    newsignal=np.asarray([])
    
    ###Cycle every second
    for i in range(0,n,step):   
        eeg=data[channel][i:i+win]     #Every second
        aux=abs(max(eeg)-min(eeg))
        if aux>threshold:
            newsignal=np.concatenate((newsignal,ones), axis=0)  
        else:
            newsignal=np.concatenate((newsignal,zeros), axis=0)
        if (i % (n//100))==0:
            print('Progress: ',(i/(n//100)))

    newsignal =newsignal[0:len(raw)]*value
    return newsignal

def upper_new(raw,name_channel,threshold,sf):  #STEP IS NOT APPLIED YET!!
    ### Extract data, sampling frequency and channels names
    data,sf,chan=raw._data,raw.info['sfreq'], raw.info['ch_names']
    data=data*1e6       # Convert Volts to uV
    print(data.shape)
    print(type(data))
    n = data.shape[1]   #samples    

   
    channel = (raw.ch_names).index(name_channel)
    print('Channel choose:',raw.ch_names[channel])
    data=data[channel][:]    
    data=data.tolist()
    #print(data[0:200])
    step=int(sf/4)
    win2=int(sf*2)
    win1=int(sf*1)
    win05=int(sf/2)
    
    dat=[0 for i in range(0,n)]

    for j in range(0,n,int(sf*30)):
        dat[j] = 4
    
    #dat=list(np.zeros((1,n)))
    #print(dat)
    ###Cycle every second
    for i in range(0,n,step):   
        eeg2=data[i:i+win2]     #Every 2 second
        eeg1=data[i:i+win1]     #Every second
        eeg05=data[i:i+win05]     #Every half second
        #print(eeg2.index(max(eeg2)))
        aux2=abs(max(data[i:i+win2])-min(data[i:i+win2]))
        aux1=abs(max(data[i:i+win1] )-min(data[i:i+win1] ))
        aux05=abs(max(data[i:i+win05])-min(data[i:i+win05]))
        if aux05>75:
            #print('entro!!!')
            #print('i= ',i)
            #print('paso',i/step)
            ind_max=data[i:i+win1].index(max(data[i:i+win1]))
            ind_max=ind_max+i
            ind_min=data[i:i+win1].index(min(data[i:i+win1]))
            ind_min=ind_min+i
            #print(ind_min)
            #print(len(dat))
            dat[ind_min]=1.25
            dat[ind_max]=1.5
        else:
            if aux1>threshold:
                ind_max=data[i:i+win1].index(max(data[i:i+win1]))
                ind_max=ind_max+i
                ind_min=data[i:i+win1].index(min(data[i:i+win1]))
                ind_min=ind_min+i
                dat[ind_min]=0.75
                dat[ind_max]=1
            else:
                if aux2>threshold:
                    ind_max=data[i:i+win2].index(max(data[i:i+win2]))
                    ind_max=ind_max+i
                    ind_min=data[i:i+win2].index(min(data[i:i+win2]))
                    ind_min=ind_min+i
                    dat[ind_min]=0.25
                    dat[ind_max]=0.5 
        if (i % (n//100))==0:
            print('Progress: ',(i/(n//100)))
    print(dat[0:100])
    return dat

def pulse(time_shape,sfreq):
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    #Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # pulse signal
    return pulso

def subtraction_eog(raw):
    eog1= raw.get_data(picks='EOG1_1') 
    eog2= raw.get_data(picks='EOG2_1')  
    sub_eog = eog1-eog2
    return sub_eog

def subtraction_emg(raw):
    emg1= raw.get_data(picks='EMG1_1') 
    emg2= raw.get_data(picks='EMG2_1')   
    sub_emg = emg1-emg2
    return sub_emg

##Re-estructure data
def re_esctructure(raw):
    data,sfreq =raw.get_data(),raw.info['sfreq']  
    time_shape = data.shape[1]
    
    sub_eog=subtraction_eog(raw)
    sub_emg =subtraction_emg(raw)

    #c3_1= raw.get_data(picks='C3_1') 
    pos_c3 = (raw.ch_names).index('C3_1')
    c3_1 = data[pos_c3,:]
    #c4_1= raw.get_data(picks='C4_1')
    pos_c4 =(raw.ch_names).index('C4_1')
    c4_1 = data[pos_c4,:]

    step=int(sfreq/4) #every half second

    new_data=data.copy()
    new_data[0]= sub_eog
    new_data[1]= sub_emg
    new_data[2]= pulse(time_shape,sfreq)
    new_data[3]= c3_1
    new_data[4]= c4_1    
    new_data[5]= upper_new(raw,'C4_1',75,step)  
    #new_data[5]= upper_than(raw,'C4_1',75,step,460) 
    #new_data[5]= Supera75(raw)
    new_data=new_data[[0,1,2,3,4,5], :]

    new_ch_names = ['EOG', 'EMG', 'Pulse', 'C3', 'C4','Supera 75']  
    #new_ch_names = ['EOG', 'EMG', 'Pulse', 'C3', 'C4','Upper'] 
    new_chtypes = ['eog'] + ['emg']+ ['misc'] + 2 *['eeg'] + ['stim'] # Recompongo los canales.
    
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
    messagebox.showinfo(message="This program allows you to tag a specific event.", title="Info")
    messagebox.askokcancelmessage=(" Was a scoring done previously with this data?")
    anotaciones = messagebox.askquestion(message=" Was a scoring done previously with this data?", title= "Anotaciones")
    if (anotaciones == 'no'):
        #messagebox.showinfo(message="Selecciona el archivo vhdr", title="Seleccion de datos")
        path = easygui.fileopenbox(title='Select VHDR file.')#selecciono la carpeta vhdr
        raw=load_brainvision_vhdr(path) 
        show_info(raw)

        #messagebox.showinfo(message="Selecciona txt con etapas de sueño", title="Seleccion de Etapas de sueño")
        path_states = easygui.fileopenbox(title='Select the hypnogram (file with extension txt).') #selecciono el txt de anotaciones anteriores
        raw,_ = set_sleep_stages(raw,path_states)
        
        raw=re_esctructure(raw)
        show_info(raw)

    elif(anotaciones == 'yes'):  
        #messagebox.showinfo(message="Selecciona el archivo fif", title="Seleccion de datos")
        path = easygui.fileopenbox(title ='Select FIF file.') #selecciono la carpeta vhdr
        raw = mne.io.read_raw_fif(path) 
  
    
    #For actual EEG/EOG/EMG/STIM data different scaling factors should be used.
    scal = dict(eeg=20e-5, eog=150e-5,emg=15e-4, misc=1e-3, stim=2)
    n_channels = 6
    order=[0,3,2,4,1,5]

    # Plot it!
    plot(raw,n_channels,scal,order)

    #Save the tagged data
    raw.annotations.save(file_name(path)+ ".txt")
    raw.save(file_name(path)+  ".fif",overwrite=True)
    
    print('Scoring was completed and the data was saved.')

if __name__ == '__main__':
    main()