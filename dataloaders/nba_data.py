import numpy as np
import os

from torch.utils.data import Dataset

class NBADataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        sample = np.load(os.path.join(
            self.root_dir, self.file_list[i]))['trajectories']
        sample[:,:,0] -= 25
        sample[:,:,0] /= 25
        sample[:,:,1] -= 50
        sample[:,:,1] /= 50
        sample[:,:,2] -= 4
        sample[:,:,2] /= 4
        return sample.reshape(200,-1)