import numpy as np
import os

from torch.utils.data import Dataset

class NBADataset(Dataset):

    def __init__(self, root_dir, idx_path=None):
        self.root_dir = root_dir
        self.idx_path = idx_path
        self.file_list = sorted(os.listdir(root_dir))
        self.idx_list = [] if self.idx_path is None else np.load(idx_path)

    def __len__(self):
        return len(self.file_list) if len(self.idx_list)==0 else len(self.idx_list)

    def __getitem__(self, i):
        if self.idx_path is None:
            file = np.load(os.path.join(
            self.root_dir, self.file_list[i]))
        else:
            idx = self.idx_list[i]
            file = np.load(os.path.join(
                self.root_dir, self.file_list[idx]))
        sample = file['trajectories']
        sample[:,:,0] -= 50
        sample[:,:,0] /= 50
        sample[:,:,1] -= 25
        sample[:,:,1] /= 25
        sample[:,:,2] -= 4
        sample[:,:,2] /= 4
        sample[:,:,3] /= 30
        sample[:,:,4] /= 30
        sample[:,:,5] /= 5

        return sample