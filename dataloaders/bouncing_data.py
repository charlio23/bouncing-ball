import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BouncingBallDataLoader(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = np.load(os.path.join(self.root_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = torch.from_numpy(self.samples[i])
        return sample

if __name__ == '__main__':
    dl = BouncingBallDataLoader('datasets/trimmed_datapoints.npy')
    print(len(dl))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False)
    print(next(iter(train_loader)).size())