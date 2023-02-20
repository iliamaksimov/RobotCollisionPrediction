import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import random


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        collision_data = []
        noncollision_data = []
        for row in self.data:
            if row[-1] == 1:
                collision_data.append(row)
            else:
                noncollision_data.append(row)
        self.data = collision_data + random.sample(noncollision_data, 2206)
        random.shuffle(self.data)
        self.data = np.array(self.data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        return {'input': np.array(self.normalized_data[idx, :-1], dtype='float32'), 'label': np.float32(self.data[idx, -1])}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        train_len = int(0.8 * len(self.nav_dataset))
        test_len = len(self.nav_dataset) - train_len
        self.train_loader, self.test_loader = data.random_split(self.nav_dataset, [train_len, test_len])
        self.train_loader = data.DataLoader(self.train_loader, batch_size=batch_size)
        self.test_loader = data.DataLoader(self.test_loader, batch_size=batch_size)

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)

    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
