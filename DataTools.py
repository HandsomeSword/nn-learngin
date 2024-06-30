import os
import pandas as pd
import numpy as np
from PIL import Image
import struct

class DataSet:
    def __init__(self, data_path, data_type='csv', transform=None):
        self.data_path = data_path
        self.data_type = data_type
        self.transform = transform
        self.data, self.labels = self._load_data()

    def _load_data(self):
        if self.data_type == 'csv':
            data_frame = pd.read_csv(self.data_path)
            labels = data_frame.iloc[:, -1].values
            data = data_frame.iloc[:, :-1].values
        else:
            data, labels = self._load_image_data()
        return data, labels

    def _load_image_data(self):
        data = []
        labels = []
        with open(self.data_path[0], 'rb') as f:
            # 读取文件的前16个字节
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            # 读取剩余的图像数据
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        
        with open(self.data_path[1], 'rb') as f:
        # 读取文件的前8个字节
            magic, num = struct.unpack('>II', f.read(8))
            # 读取剩余的标签数据
            labels = np.fromfile(f, dtype=np.uint8)
        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= len(self.dataset):
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.current_idx += self.batch_size
        data, labels = zip(*batch)
        return np.array(data), np.array(labels)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
