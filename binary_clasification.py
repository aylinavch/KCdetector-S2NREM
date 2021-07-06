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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from random import shuffle


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

def align_KC_nofilter(start, end, data, sf):
    d = 2 #in seconds
    dur = d * sf
    KC_numpy = data[start:end]
    c = center(KC_numpy)
    cent = c + start
    mid_dur = (2 - (start - end)) // 2
    starting = cent - int(sf*d/2)
    ending = starting + d * sf

    a_KC = data[int(starting):int(ending)]

    return a_KC, len(a_KC)

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

def getKCs_nofilter(filt, name_channel = 'C4_1', sf = 200, pathKC = ''): #Return a vector with only KCs labeled (everything else is zero)
    
    timeseries = []
    
    for j in range(0, len(filt)):
        print('filt:',  filt[j][0])
        data = filt[j][0].tolist()
        path_KC = pathKC[j]
        filetxt = open(path_KC, 'r')
        print('S', path_KC[41:43])

        for x in filetxt:
            if x[-3] == 'K' and x[-2] == 'C':
                vector, cantKC = extract_td(x)  #vector[0,0] = position of KC // vector[0,1] = duration of KC
                start = int(vector[0,0]*sf)
                end = int((vector[0,0] + vector[0,1])*sf)
                a_KC, leng = align_KC_nofilter(start, end, data, sf)
                timeseries.append(a_KC)
        
        filetxt.close()
    
    return timeseries

def getKCs(filt, name_channel = 'C4_1', sf = 200, pathKC = ''): #Return a vector with only KCs labeled (everything else is zero)
    
    timeseries = []
    
    for j in range(0, len(filt)):
        data = filt[j][0]
        path_KC = pathKC[j]
        filetxt = open(path_KC, 'r')
        print('S', path_KC[41:43])

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
            #pos_inic_noKC = [1005,1180,1220,1320,1360,1485,1585,1645,1730,1860,1945,2020,2130,2215]
            pos_inic_noKC = [1005,1180,1255.5,1288,1300.5,1331.5,1507.5,1529,1655,1743,1770,1789,1801,1900.5]
            data = filt[1][0]

        elif sN == 17:
            #pos_inic_noKC = [910,920,995,1050,1140,1175,1235,1275,1330,1375,1445,1525,1565,1660,1730,2033]
            pos_inic_noKC = [910,970,979.5,988.5,1145,1160.5,1230.5,1251,1320,1336,1436.5,1444,1466,1504,1506,1534]
            data = filt[2][0]

        elif sN == 21:
            #pos_inic_noKC = [575,605,690,740,805,845,920,955,1015,1090,1160,1200,1255,1260,2715,2770,2805,2860,2875,2905]
            pos_inic_noKC = [635.5,663.5,854.5,740,878.5,962,980,1006,1052.5,1055.5,1085.5,1096.5,1148.5,1200,1204.5,1213,1219,1234,1240.5,1258.5]
            data = filt[3][0]

        elif sN == 28:
            pos_inic_noKC = [293,318,378.5,505,582.5,666.5,793.5]
            data = filt[4][0]

        elif sN == 36:
            #pos_inic_noKC = [435,500,620,750,820.5,1065,1180,1270,1340,1510,1660,1825,1900,1980,2018,2100,2130,2180,2420,2600,3035,3350,3540,3665,3875,4100]
            pos_inic_noKC = [542,784.5,804,1083,1117.5,1129,1182,1211,1273.5,1295.5,1305.5,1324.5,1344.5,1434,1493.5,1509,1589,1636.5,1664,1740.5,1798,1817,1927,2203,2221,2309]
            data = filt[5][0]
        
        elif sN == 37:
            pos_inic_noKC = [1776.5,2055]
            data = filt[6][0]

        elif sN == 40:
            #pos_inic_noKC = [755,925,945,1040,1090,1160,1200,1255,1295,1370,1495,1840]
            pos_inic_noKC = [767,774,921,960,1021,1044,1160.5,1187,1329,1364,1744,1844]
            data = filt[7][0]
        else:
            print('Do not use S',sN, ' because there are no KCs enough to make a pipeline')

        for start in pos_inic_noKC:
        
            noKCs.append(data[int(start):int(start+2*sf)]) #tolist()

    return noKCs

def clasification(x , name_channel = 'C4_1',sf = 200, path_KC = ''):
    
    yesKCs = getKCs(x, name_channel = 'C4_1', sf = sf, pathKC = path_KC)
    noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC

    print('long yesKCs', len(yesKCs))
    print('long noKCs', len(noKCs))

    y = np.concatenate((y_yesKCs,y_noKCs))
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    #scores = cross_validate(clf, X, y, cv = cv)
    #print('Scores with 10-CV: ', scores)
    #test = [yesKCs[0], noKCs[0]]
    #print(clf.predict(test))
    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    #clf.fit(X_train, y_train)
    #print('SCORE:', clf.score(X_test, y_test))
    #svc_disp = plot_roc_curve(clf, X, y)
    #plt.show()

    return clf

def clasification_random(x , name_channel = 'C4_1',sf = 200, path_KC = ''):
    
    yesKCs = getKCs(x, name_channel = 'C4_1', sf = sf, pathKC = path_KC)
    noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

    print('long yesKCs', len(yesKCs))
    print('long noKCs', len(noKCs))

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC

    y = np.concatenate((y_yesKCs,y_noKCs))
    print(y)
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)

    np.random.shuffle(y)
    print(y)
    scores = cross_val_score(clf, X, y, cv=cv)
    #scores = cross_validate(clf, X, y, cv = cv)
    print('Scores with 10-CV: ', scores)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    #clf.fit(X_train, y_train)
    #print('SCORE:', clf.score(X_test, y_test))
    #svc_disp = plot_roc_curve(clf, X, y)
    #plt.show()

    return clf

def clasification_nofilter(x , name_channel = 'C4_1',sf = 200, path_KC = ''):
    
    yesKCs = getKCs_nofilter(x, name_channel = 'C4_1', sf = sf, pathKC = path_KC)
    noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

    print('long yesKCs', len(yesKCs))
    print('long noKCs', len(noKCs))

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs)) # 0 --> no KC

    y = np.concatenate((y_yesKCs,y_noKCs))
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    cv = ShuffleSplit(n_splits=10, test_size=0.5, random_state=0)
    scores = cross_val_score(clf, X, y, cv=cv)
    #scores = cross_validate(clf, X, y, cv = cv)
    print('Scores with 10-CV: ', scores)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    #clf.fit(X_train, y_train)
    #print('SCORE:', clf.score(X_test, y_test))
    #svc_disp = plot_roc_curve(clf, X, y)
    #plt.show()

    return clf

def multiclasification(x, name_channel = 'C4_1', sf = 200, path_KC = ''):

    yesKCs = getKCs(x, name_channel = 'C4_1', sf = sf, pathKC = path_KC)
    noKCs = get_noKCs(x, name_channel = 'C4_1', sf = sf)

    y_yesKCs = np.ones(len(yesKCs)) # 1 --> yes KC
    y_noKCs = np.zeros(len(noKCs))  # 0 --> no KC

    y = np.concatenate((y_yesKCs,y_noKCs))
    X = np.concatenate((np.array(yesKCs), np.array(noKCs)))

    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    
    reorder = np.random.permutation(X.shape[0])

    ## SVM
    model_SVM = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True, random_state = 42))
    scores_SVM = cross_val_score(model_SVM, X, y, cv=cv)
    print('SVM cv scores: ', scores_SVM)
    model_SVM.fit(X_train, y_train)
    probs_SVM = model_SVM.predict_proba(X_test)
    kr_probs = probs_SVM[:, 1]
    SVM_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', SVM_auc)

    ## kNN
    #model_kNN = KNeighborsClassifier(n_neighbors=5)
    #model_kNN.fit(trainingdata,traininglabels)
    model_kNN = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    scores_kNN = cross_val_score(model_kNN, X, y, cv=cv)
    print('kNN cv scores: ', scores_kNN)
    model_kNN.fit(X_train, y_train)
    probs_kNN = model_kNN.predict_proba(X_test)
    kr_probs = probs_kNN[:, 1]
    kNN_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', kNN_auc)

    ## LogReg
    #logisticRegr = LogisticRegression()
    #logisticRegr.fit(trainingdata,traininglabels)
    model_LogReg = make_pipeline(StandardScaler(),  LogisticRegression())
    scores_LogReg = cross_val_score(model_LogReg, X, y, cv=cv)
    print('LogReg cv scores: ', scores_LogReg)
    model_LogReg.fit(X_train, y_train)
    probs_LogReg = model_LogReg.predict_proba(X_test)
    kr_probs = probs_LogReg[:, 1]
    LogReg_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', LogReg_auc)

    ## LDA
    #model = LinearDiscriminantAnalysis()
    #ldaf.fit(trainingdata,traininglabels)
    model_LDA = make_pipeline(StandardScaler(),  LinearDiscriminantAnalysis())
    scores_LDA = cross_val_score(model_LDA, X, y, cv=cv)
    print('LDA cv scores: ', scores_LDA)
    model_LDA.fit(X_train, y_train)
    probs_LDA = model_LDA.predict_proba(X_test)
    kr_probs = probs_LDA[:, 1]
    LDA_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', LDA_auc)


    ## Decision Tree
    #dt = DecisionTreeClassifier(random_state=0)
    #dt.fit(trainingdata, traininglabels)
    model_DT = make_pipeline(StandardScaler(),  DecisionTreeClassifier(random_state=0))
    scores_DT = cross_val_score(model_DT, X, y, cv=cv)
    print('Decision Tree cv scores: ', scores_DT)
    model_DT.fit(X_train, y_train)
    probs_DT = model_DT.predict_proba(X_test)
    kr_probs = probs_DT[:, 1]
    DT_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', DT_auc)


    ## Random Forest
    #frs = RandomForestClassifier(n_estimators=100, random_state=0)
    #frs.fit(trainingdata, traininglabels)
    model_RF = make_pipeline(StandardScaler(),  RandomForestClassifier(n_estimators=100, random_state=0))
    scores_RF = cross_val_score(model_RF, X, y, cv=cv)
    print('Random Forest cv scores: ', scores_RF)
    model_RF.fit(X_train, y_train)
    probs_RF = model_RF.predict_proba(X_test)
    kr_probs = probs_RF[:, 1]
    RF_auc = roc_auc_score(X_test, y_test)
    print('AUC = ', RF_auc)

    """
# predict probabilities
sr_probs = clf.predict_proba(testdata)
# keep probabilities for the positive outcome only
sr_probs = sr_probs[:, 1]


# predict probabilities
kr_probs = model.predict_proba(testdata)
# keep probabilities for the positive outcome only
kr_probs = kr_probs[:, 1]


# predict probabilities
lr_probs = logisticRegr.predict_proba(testdata)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

# predict probabilities
dr_probs = ldaf.predict_proba(testdata)
# keep probabilities for the positive outcome only
dr_probs = dr_probs[:, 1]

# predict probabilities
dt_probs = dt.predict_proba(testdata)
# keep probabilities for the positive outcome only
dt_probs = dt_probs[:, 1]

# predict probabilities
rf_probs = frs.predict_proba(testdata)
# keep probabilities for the positive outcome only
rf_probs = rf_probs[:, 1]


# calculate scores
ns_auc = roc_auc_score(testlabels, ns_probs)
sr_auc = roc_auc_score(testlabels, sr_probs)
kr_auc = roc_auc_score(testlabels, kr_probs)
lr_auc = roc_auc_score(testlabels, lr_probs)
dr_auc = roc_auc_score(testlabels, dr_probs)
dt_auc = roc_auc_score(testlabels, dt_probs)
rf_auc = roc_auc_score(testlabels, rf_probs)

# summarize scores
print('Trivial: ROC AUC=%.3f' % (ns_auc))
print('SVM: ROC AUC=%.3f' % (sr_auc))
print('kNN: ROC AUC=%.3f' % (kr_auc))
print('LogReg: ROC AUC=%.3f' % (lr_auc))
print('LDA: ROC AUC=%.3f' % (dr_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))
print('Random Forest: ROC AUC=%.3f' % (rf_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testlabels, ns_probs)
sr_fpr, sr_tpr, _ = roc_curve(testlabels, sr_probs)
kr_fpr, kr_tpr, _ = roc_curve(testlabels, kr_probs)
lr_fpr, lr_tpr, _ = roc_curve(testlabels, lr_probs)
dr_fpr, dr_tpr, _ = roc_curve(testlabels, dr_probs)
dt_fpr, dt_tpr, _ = roc_curve(testlabels, dt_probs)
rf_fpr, rf_tpr, _ = roc_curve(testlabels, rf_probs)
    """

def pulse(time_shape,sfreq): #To generate 0.5 sec x-axes
    t = np.linspace(1, round(time_shape/sfreq), time_shape, endpoint=False)    # Create artificial signal with a 0.5 sec pulse
    pulso = signal.square(2 * np.pi * 1 * t) # Pulse signal
    return pulso

############# Main function ##########################################################
def main():  # Wrapper function
    messagebox.showinfo(message ="This program allows you to tag a specific event.", title="Info")

    ## S11
    print('Cargando datos del Sujeto 11')
    path11 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS11.vhdr'
    raw11 = load_brainvision_vhdr(path11) 
    show_info(raw11)
    #path_states11 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS11.txt'
    #raw11,_ = set_sleep_stages(raw11,path_states11)
    path_KC11 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS11_2020September28_19-48.txt'       
    #raw_end11 = re_esctructure(raw11, path_KC11)
    #show_info(raw_end11)
    data11,sfreq11 =raw11.get_data(),raw11.info['sfreq']

    ## S12
    print('Cargando datos del Sujeto 12')
    path12 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS12.vhdr'
    raw12 = load_brainvision_vhdr(path12) 
    show_info(raw12)
    #path_states12 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS12.txt'
    #raw12,_ = set_sleep_stages(raw12,path_states12)
    path_KC12 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS12_2020September28_20-01.txt'     
    #raw_end12 = re_esctructure(raw12, path_KC12)
    #show_info(raw_end12)
    data12,sfreq12 =raw12.get_data(),raw12.info['sfreq']

    ## S17
    print('Cargando datos del Sujeto 17')
    path17 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS17.vhdr'
    raw17 = load_brainvision_vhdr(path17) 
    show_info(raw17)
    #path_states17 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS17.txt'
    #raw17,_ = set_sleep_stages(raw17,path_states17)
    path_KC17 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS17_2020September28_20-14.txt'       
    #raw_end17 = re_esctructure(raw17, path_KC17)
    #show_info(raw_end17)
    data17,sfreq17 =raw17.get_data(),raw17.info['sfreq']

    ## S21
    print('Cargando datos del Sujeto 21')
    path21 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS21.vhdr'
    raw21 = load_brainvision_vhdr(path21) 
    show_info(raw21)
    #path_states21 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS21.txt'
    #raw21,_ = set_sleep_stages(raw21,path_states21)
    path_KC21 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS21_2020September28_20-29.txt'       
    #raw_end21 = re_esctructure(raw21, path_KC21)
    #show_info(raw_end21)
    data21,sfreq21 =raw21.get_data(),raw21.info['sfreq']

    ## S28
    print('Cargando datos del Sujeto 28')
    path28 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS28.vhdr'
    raw28 = load_brainvision_vhdr(path28) 
    show_info(raw28)
    path_states28 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS28.txt'
    #raw28,_ = set_sleep_stages(raw28,path_states28)
    path_KC28 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS28_2020September28_20-40.txt'       
    #raw_end28 = re_esctructure(raw28, path_KC28)
    #show_info(raw_end28)
    data28,sfreq28 =raw28.get_data(),raw28.info['sfreq']

    ## S36
    print('Cargando datos del Sujeto 36')
    path36 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS36.vhdr'
    raw36 = load_brainvision_vhdr(path36) 
    show_info(raw36)
    path_states36 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS36.txt'
    #raw36,_ = set_sleep_stages(raw36,path_states36)
    path_KC36 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS36_2020September29_14-00.txt'       
    #raw_end36 = re_esctructure(raw36, path_KC36)
    #show_info(raw_end36)
    data36,sfreq36 =raw36.get_data(),raw36.info['sfreq']

    ## S37
    print('Cargando datos del Sujeto 37')
    path37 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS37.vhdr'
    raw37 = load_brainvision_vhdr(path37) 
    show_info(raw37)
    path_states37 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS37.txt'
    #raw37,_ = set_sleep_stages(raw37,path_states37)
    path_KC37 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS37_2020September29_11-01.txt'       
    #raw_end37 = re_esctructure(raw37, path_KC37)
    #show_info(raw_end37)
    data37,sfreq37 =raw37.get_data(),raw37.info['sfreq']    

    ## S40
    print('Cargando datos del Sujeto 40')
    path40 = 'C:/Users/aylin/PFC/Señales/RCnew/ExpS40.vhdr'
    raw40 = load_brainvision_vhdr(path40) 
    show_info(raw40)
    path_states40 = 'C:/Users/aylin/PFC/Señales/TXT/ExpS40.txt'
    #raw40,_ = set_sleep_stages(raw40,path_states40)
    path_KC40 = 'C:/Users/aylin/PFC/Etiquetas OK/.TXT/ExpS40_2020September29_11-10.txt'       
    #raw_end40 = re_esctructure(raw40, path_KC40)
    #show_info(raw_end40)
    data40,sfreq40 =raw40.get_data(),raw40.info['sfreq']

    print('------------------------------------------------------------------------')

    data = [data11, data12, data17, data21, data28, data36, data37, data40]
    raw = [raw11, raw12, raw17, raw21, raw28, raw36, raw37, raw40]
    path_KC = [path_KC11,path_KC12,path_KC17,path_KC21,path_KC28,path_KC36,path_KC37,path_KC40]
    
    filt = []

    print('Filtrando todas las señales')

    for num in range(0,len(raw)):
        r = raw[num]
        f = filtering(raw = r, name_channel = 'C4_1', sf = 200)
        filt.append(f)
        print('............................................')

    #print('Comenzando a entrenar el modelo con todos los datos')
    #print()
    #print('Sin filtrar la señal:')
    #clasification_nofilter(data, path_KC = path_KC)
    #print()
    #print('Filtrando la señal:')
    #clasification(filt, path_KC = path_KC)
    #print()
    #print('Permutando los labels')
    #clasification_random(filt, path_KC = path_KC)

    multiclasification(filt, path_KC = path_KC)

    print('------------------------------------------------------------------------')


if __name__ == '__main__':
    main()