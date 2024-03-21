import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self,
                 base_path: str = '../data/synthesized/',
                 image_folder: str = 'base/',
                 target_dim: Tuple[int] = (256, 256)):
        '''
        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.
        '''
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))

        self.image_by_patient = []

        for folder in all_image_folders:
            paths = sorted(glob('%s/*.png' % (folder)))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 3


class SyntheticSubset(SyntheticDataset):

    def __init__(self,
                 main_dataset: SyntheticDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'array']):
        '''
        A subset of SyntheticDataset.

        In SyntheticDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of SyntheticSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_subsequences = []
        for image_list in self.image_by_patient:
            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])

    def __len__(self) -> int:
        if self.return_format == 'all_subsequences':
            # If we return all subsequences of images per patient...
            return len(self.all_subsequences)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

            images = normalize_image(images)

        return images, timestamps


def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                             code=cv2.COLOR_BGR2RGB), target_dim))
    else:
        image = np.array(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                         code=cv2.COLOR_BGR2RGB))

    if normalize:
        # Normalize image.
        image = normalize_image(image)

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image

def normalize_image(image: np.array) -> np.array:
    image = (image / 255 * 2) - 1
    return image

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = path.split('time_')[1].replace('.png', '')
    # Shall be 3 digits
    assert len(time) == 3
    time = float(time)
    return time
