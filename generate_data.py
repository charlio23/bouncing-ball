import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from datasets.bouncing_ball import BouncingBall2D

def generate_and_save(root_path, n_samples, start_seed, train=True):
    path = os.path.join(root_path, 'train' if train else 'test')
    if not os.path.exists(path):
        os.makedirs(path)
    """
    for i in tqdm(range(n_samples)):
        sample = lin_data.sample_sequences(sequence_len=seq_len, number_of_sequences=1, seed=i+start_seed)[0]
        filename = "{0:05d}".format(i)
        np.savez(os.path.join(path, filename), evolution=sample['evolution'], graph=sample['graph'])
    return path
    """
    for i in tqdm(range(n_samples)):
        failed = True
        while(failed):
            game = BouncingBall2D()
            positions, images, background = game.run()
            if positions is not None:
                failed = False
                filename = "{0:05d}".format(i)
                np.savez(os.path.join(path, filename), positions=positions, images=images, background=background)

    return path

if __name__ == "__main__":

    DEFAULT_DATASETS_ROOT = 'datasets/'
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--name', action='store', nargs=1, required=True, help='The dataset name.'
    )
    parser.add_argument(
        '--ntrain', action='store', required=False, type=int, default=100,
        help='Number of training sample to generate.'
    )
    parser.add_argument(
        '--ntest', action='store', required=False, type=int, default=100,
        help='Number of test samples to generate.'
    )
    parser.add_argument(
        '--datasets-root', action='store', nargs=1, required=False, type=str,
        help=f'Root of the datasets folder in which the dataset will be stored. If not specified, '
             f'{DEFAULT_DATASETS_ROOT} will be used as default.'
    )

    _args = parser.parse_args()
    # Extract environment parameters
    EXP_NAME = _args.name[0]
    N_TRAIN_SAMPLES = _args.ntrain
    N_TEST_SAMPLES = _args.ntest


    # Get dataset output path
    dataset_root = DEFAULT_DATASETS_ROOT if _args.datasets_root is None else _args.datasets_root[0]
    dataset_root = os.path.join(dataset_root, EXP_NAME)

    # Generate train samples
    print("Generating train data...")
    _train_path = generate_and_save(
        root_path=dataset_root,
        n_samples=N_TRAIN_SAMPLES, 
        start_seed=0, train=True)

    # Generate test samples
    print("Generating test data...")
    _test_path = None
    if N_TEST_SAMPLES > 0:
        _test_path = generate_and_save(
        root_path=dataset_root,
        n_samples=N_TEST_SAMPLES, 
        start_seed=N_TRAIN_SAMPLES, train=False)

    print("Data generation finished!")