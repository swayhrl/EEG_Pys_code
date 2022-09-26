from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import numpy as np
import math
import os
from scipy.signal import stft
from scipy.fftpack import fft,ifft
from warnings import filterwarnings
filterwarnings('ignore')
fStart = [0.5, 4, 8, 14, 30]
fEnd = [4, 8, 14, 30, 100]
fs = 256 #采样
window = 'hann'
# frame长度
Hm = 4000

def Process2(data_path1,data_path2):
    dict = {'fear_1':'fe1','anxiety_1':'ax1','happy_1':'ha1','angry_1':'ag1','helpless_1':'hl1',
            'fear_2':'fe2','anxiety_2':'ax2','happy_2':'ha2','angry_2':'ag2','helpless_2':'hl2',
            'fear_3':'fe3','anxiety_3':'ax3','happy_3':'ha3','angry_3':'ag3','helpless_3':'hl3'}
    data_all = loadmat(data_path1)
    for key in data_all.keys():
        pass
    data1 = data_all[key].item()
    data1 = data1[15]
    temp = data_path2.split('.')[0]
    temp = temp.split('_',maxsplit=2)[-1]
    key = dict[temp]
    data2 = loadmat(data_path2)
    data2_1 = data2['ecg_'+key]
    data2_2 = data2['eda_'+key]
    data2_3 = data2['emg_'+key]
    data2_4 = data2['emgc_'+key]
    data2_5 = data2['hr_'+key]
    data2_6 = data2['ppg_'+key]
    data2_7 = data2['pr_'+key]
    data2_8 = data2['skt_'+key]
    data2 = np.hstack([data2_1, data2_2, data2_3, data2_4, data2_5, data2_6, data2_7, data2_8])
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)  # 归一化
    result1 = []
    result2 = []
    for channel in range(data1.shape[0]):  # 遍历64通道
        temp = []
        for band_index, band in enumerate(fStart):  # 遍历5种频段
            b, a = signal.butter(1, [fStart[band_index] / fs, fEnd[band_index] / fs], 'bandpass')  # 配置滤波器 4 表示滤波器的阶数
            filtedData = signal.filtfilt(b, a, data1[channel])  # data为要过滤的信号,filtedData为得到的过滤信号
            # STFT
            f, t, Z = stft(filtedData, fs=fs, window=window, nperseg=Hm, noverlap=0)
            temp.append(np.mean(np.abs(Z)))
        result1.append(temp)
    for channel in range(data2.shape[1]):  # 遍历8通道
        temp = []
        for band_index, band in enumerate(fStart):  # 遍历5种频段
            b, a = signal.butter(1, [fStart[band_index] / fs, fEnd[band_index] / fs], 'bandpass')  # 配置滤波器 4 表示滤波器的阶数
            filtedData = signal.filtfilt(b, a, data2[:,channel])  # data为要过滤的信号,filtedData为得到的过滤信号
            # STFT
            f, t, Z = stft(filtedData, fs=fs, window=window, nperseg=Hm, noverlap=0)
            temp.append(np.mean(np.abs(Z)))
        result2.append(temp)
    results=np.vstack([result1,result2]) #得到最终的包含多模态频域信息关联的融合特征 72*5
    return np.array(results,dtype=np.float32)