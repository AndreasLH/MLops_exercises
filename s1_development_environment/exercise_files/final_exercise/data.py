import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def mnist_():
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 
    return train, test

def mnist():
    """Pytorch version of the mnist dataset
    
        returns a train and test loader"""
    # exchange with the corrupted mnist dataset
    imgs = []
    labels = []
    for i in range(5):
        train = np.load(f'data/corruptmnist/train_{i}.npz')
        images_ = train['images']
        labels_ = train['labels']
        imgs.append(images_)
        labels.append(labels_)
    images = np.concatenate(imgs)
    labels = np.concatenate(labels)
    
    test = np.load('data/corruptmnist/test.npz')

    class Train_dataset(Dataset):
        def __init__(self):
            self.data = torch.from_numpy(images).view(-1,1,28,28)
            self.label = torch.from_numpy(labels)
        def __getitem__(self, index):
            return self.data[index].float(), self.label[index]
        def __len__(self):
            return len(self.data)

    train_dataset = Train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    class Test_dataset(Dataset):
        def __init__(self):
            self.data = torch.tensor(test['images']).view(-1,1,28,28)
            self.label = torch.tensor(test['labels'])
        def __getitem__(self, index):
            return self.data[index].float(), self.label[index]
        def __len__(self):
            return len(self.data)

    Test_dataset = Test_dataset()
    test_loader = DataLoader(Test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader