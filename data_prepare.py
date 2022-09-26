from Process.process1 import *
from Process.process2 import *
from Process.process3 import *
import os
from warnings import filterwarnings
filterwarnings('ignore')
from torch.utils import data

def Data_path():
    label_dict = {'angry1':0,'angry2':0,'angry3':0,'anxiety1':1,'anxiety2':1,'anxiety3':1,'fear1':2,'fear2':2,'fear3':2,'happy1':3,'happy2':3,'happy3':3,'helpless1':4,'helpless2':4,'helpless3':4}
    eeg_root_dir = './EEG'
    pys_root_dir = './Pys'
    EGG_MAT_list =[]
    PYS_MAT_list =[]
    LABEL_list=[]
    subdir_list = os.listdir(eeg_root_dir)
    for sub_dir in subdir_list:
        eeg_sub_path = os.path.join(eeg_root_dir,sub_dir)
        pys_sub_path = os.path.join(pys_root_dir,sub_dir)
        mat_list = os.listdir(eeg_sub_path) #得到具体文件名
        for mat in mat_list:
            egg_mat = os.path.join(eeg_sub_path,mat)
            EGG_MAT_list.append(egg_mat)
            pys_mat = os.path.join(pys_sub_path,mat) #定位到具体文件
            PYS_MAT_list.append(pys_mat)
            mat = mat.split('.')[0]
            label1 = mat.split('_')[2] #根据字典找到对应标签
            label2 = mat.split('_')[3]
            LABEL_list.append(label_dict[label1+label2])
    return  EGG_MAT_list,PYS_MAT_list,LABEL_list


class MyDataset(data.Dataset):
    def __init__(self, eggmat_paths, pysmat_paths,labels):
        self.eggmat_paths = eggmat_paths
        self.pysmat_paths = pysmat_paths
        self.labels = labels

    def __getitem__(self, index):
        data1 = Process1(self.eggmat_paths[index])
        data2 = Process2(self.eggmat_paths[index], self.pysmat_paths[index])
        data3 = Process3(self.eggmat_paths[index], self.pysmat_paths[index])
        label = self.labels[index]
        return data1,data2,data3,label

    def __len__(self):
        return len(self.eggmat_paths)





