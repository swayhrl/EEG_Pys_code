# EEG_Pys_code
## 如何运行
将文件结构按照文件夹结构.txt放置。
当训练时，将`data_prepare.py`中的`eeg_root_dir`设置为`./train_EEG`,`pys_root_dir`设置为`./train_Pys`,将`main.py`中的`mode`设置为`train`(默认为此，可不变)
```bash
python3 data_prepare.py
python3 main.py
```
在验证时，将`data_prepare.py`中的`eeg_root_dir`设置为`./test_EEG`,`pys_root_dir`设置为`./test_Pys`,将`main.py`中的`mode`设置为`test`
```bash
python3 data_prepare.py
python3 main.py
```

## 数据集下载
百度云盘链接：https://pan.baidu.com/s/1LvsO2Fq_JfWLu9RscD1gqw?pwd=1234 
提取码：1234

## 验证结果
在验证集中达到86.11%的成绩!![1e59dbd43317b2bca105a1123893aae](https://user-images.githubusercontent.com/56393103/192287400-7a295329-0554-4c99-98d3-f4eb75fd1a76.png)

