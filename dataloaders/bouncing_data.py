import os
import numpy as np
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt, animation
import cv2
from glob import glob

class MissingBallDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_labels = self.img_dir
        try:
            self.img_filenames = glob(f"{self.img_dir}/*")
        except:
            raise ValueError("img_dir incorrect")

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_path = self.img_filenames[idx]
         
        stored_obj = np.load(img_path)
        img = stored_obj["images"][:,np.newaxis]
        # missing_frames = stored_obj["missing_frames"]
        missing_mask = stored_obj["missing_frames"]
        return img, missing_mask

class BouncingBallDataLoader(Dataset):

    def __init__(self, root_dir, images=True):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.images = images
        if images:
            self.key = 'images'
        else:
            self.key = 'positions'

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        sample = np.load(os.path.join(
            self.root_dir, self.file_list[i]))
        pos = sample['positions']
        pos = pos[:,:2]/128.0 - 1.0
        im = sample['images']
        if len(im.shape) == 3:
            im = im[:,np.newaxis,:,:].astype(float)/255.0
        else:
            im = im.transpose((0,3,1,2))/255.0
        return im, pos


def visualize_rollout(rollout, interval=50, show_step=False, save=False):
    """Visualization for a single sample rollout of a physical system.
    Args:
        rollout (numpy.ndarray): Numpy array containing the sequence of images. It's shape must be
            (seq_len, height, width, channels).
        interval (int): Delay between frames (in millisec).
        show_step (bool): Whether to draw the step number in the image
    """
    fig = plt.figure()
    img = []
    for i, im in enumerate(rollout):
        if show_step:
            black_img = np.zeros(list(im.shape))
            cv2.putText(
                black_img, str(i), (0, 30), fontScale=0.22, color=(255, 255, 255), thickness=1,
                fontFace=cv2.LINE_AA)
            res_img = (im + black_img / 255.) / 2
        else:
            res_img = im
        img.append([plt.imshow(res_img, animated=True)])
    ani = animation.ArtistAnimation(fig,
                                    img,
                                    interval=interval,
                                    blit=True,
                                    repeat_delay=100)
    if save:
        writergif = animation.PillowWriter(fps=30)
        ani.save('dataloaders/bouncing_sequence.gif', writergif)
    plt.show()


if __name__ == '__main__':
    dl = BouncingBallDataLoader('datasets/bouncing_ball/train', False)
    print(len(dl))
    train_loader = torch.utils.data.DataLoader(dl, batch_size=1, shuffle=False)
    sample = next(iter(train_loader))
    print(torch.max(sample))
    print(torch.min(sample))
    print(sample.size())
    #visualize_rollout(sample[0], save=True)