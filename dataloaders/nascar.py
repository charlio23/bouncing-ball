import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt, animation


class NascarDataLoader(Dataset):

    def __init__(self, root_file, seq_len=50):
        
        self.root_file = root_file
        self.seq_len = seq_len
        file = np.load(root_file)
        self._y = file['y']
        self._x = file['x']
        self._z = file['z']
        
    def __len__(self):
        return len(self._y)//self.seq_len

    def __getitem__(self, i):
        y = self._y[i*self.seq_len:(i+1)*self.seq_len]
        x = self._x[i*self.seq_len:(i+1)*self.seq_len]
        z = self._z[i*self.seq_len:(i+1)*self.seq_len]
        return y, x, z


if __name__ == '__main__':
    dl = NascarDataLoader('/data2/users/cb221/nascar.npz', 50)
    print(len(dl))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=4, shuffle=False)
    y, x, z = next(iter(train_loader))
    print(y.size())
    print(x.size())
    print(z.size())

