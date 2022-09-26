from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
from scipy import signal
import numpy as np
import math
from scipy.signal import stft
from warnings import filterwarnings
filterwarnings('ignore')
fStart = [0.5, 4, 8, 14, 30]
fEnd = [4, 8, 14, 30, 100]
fs = 256 #采样
window = 'hann'
# frame长度
Hm = 4000

def compute_DE(signal):
    variance = np.var(signal, ddof=1)  # 求得方差
    return math.log(2 * math.pi * math.e * variance) / 2  # 微分熵求取公式

def Process1(data_path):
    data_all = loadmat(data_path)
    for key in data_all.keys():
        pass
    data_all = data_all[key].item()
    data = data_all[15] #得到要处理的特征
    # scaler = MinMaxScaler()
    # data = scaler.fit_transform(data)  # 归一化
    results = []
    temp=[]
    result = []
    for channel in range(data.shape[0]): #遍历64通道
        for band_index, band in enumerate(fStart): #遍历5频段
            b, a = signal.butter(1, [fStart[band_index] / fs, fEnd[band_index] / fs], 'bandpass')  # 配置滤波器 4 表示滤波器的阶数
            filtedData = signal.filtfilt(b, a, data[channel])  # data为要过滤的信号,filtedData为得到的过滤信号
            # STFT
            f,t,Z= stft(filtedData, fs=fs, window=window, nperseg=Hm,noverlap=0)
            Z = np.abs(Z)
            # 计算微分熵
            for de_index in range(0, Z.shape[0] - fs, fs):
                de = compute_DE(Z[de_index: de_index + fs])
                temp.append(de)
            result.append(temp)
            temp = []
        results.append(result)
        #print("results_3",len(results))
        result=[]
    results = np.array(results)
    results = np.transpose(results,(2,0,1))
    return np.array(results,dtype=np.float32) #输出为7*(64*5)
