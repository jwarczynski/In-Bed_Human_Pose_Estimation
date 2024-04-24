from src.visualization import BODY_PARTS

from easydict import EasyDict
import matplotlib.pyplot as plt
from cv2 import imread
from scipy.io import loadmat
import pandas as pd
import glob
import os

from torch.utils.data import DataLoader
import torch

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


class HumanPoseDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.img_paths = self.__get_modality_img_pairs()
        if train:
            self.ground_truth_df = self.__get_ground_truth()

    def __get_modality_img_pairs(self):
        img_paths = []
        for subject_dir in glob.glob(f'{self.root_dir}/*'):
            if self.train and glob.glob(f'{subject_dir}/*.mat') == []:
                # logger.info(f'Subject {subject_dir} does not have ground truth data. Skipping...')
                continue
            modality_dirs = glob.glob(f'{subject_dir}/*/')
            modality_img_pairs = zip(*[glob.glob(f'{modality}/*/*.png') for modality in modality_dirs])
            img_paths.extend(modality_img_pairs)
        return img_paths

    def __get_ground_truth(self):
        col_names = ['Subject_id', 'Image_id', 'Modality']
        col_names.extend(BODY_PARTS)
        gt_df = pd.DataFrame(columns=col_names)

        for subject_dir in os.listdir(self.root_dir):
            gt_matrices = glob.glob(f'{os.path.join(self.root_dir, subject_dir)}/*.mat')
            modalities = [gt_file.split('.mat')[0].split('_')[-1] for gt_file in gt_matrices]
            subject_gt = get_ground_truth_for_subject(gt_matrices, modalities, subject_dir)
            gt_df = pd.concat([gt_df, pd.DataFrame(subject_gt, columns=col_names)], ignore_index=True)

        gt_df.set_index(['Subject_id', 'Image_id', 'Modality'], inplace=True)
        return gt_df

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_pair_paths = self.img_paths[idx]
        indexes = extract_indexes(img_pair_paths)
        images = [imread(img_path) for img_path in img_pair_paths]

        if self.transform:
            images = (self.transform(img) for img in images)

        images = EasyDict({path.split('\\')[-3]: img for path, img in zip(img_pair_paths, images)})
        labels = []
        if self.train:
            labels = EasyDict({idx[-1]: self.ground_truth_df.loc[idx].values.tolist() for idx in indexes})

        return images, labels


def get_ground_truth_for_subject(gt_matrices, modalities, subject_dir):
    subject_gt = []
    for gt_matrix, modality in zip(gt_matrices, modalities):
        gt = loadmat(gt_matrix)
        for i in range(gt['joints_gt'].shape[2]):
            x_parts_locations = tuple(gt['joints_gt'][0, :, i])
            y_parts_locations = tuple(gt['joints_gt'][1, :, i])
            if_part_occluded = tuple(gt['joints_gt'][2, :, i])
            parts_gt = tuple(zip(x_parts_locations, y_parts_locations, if_part_occluded))

            row = [int(subject_dir), i + 1, modality]
            row.extend(parts_gt)

            subject_gt.append(row)

    return subject_gt


def extract_index(img_path):
    subject_id = int(img_path.split('\\')[-4])
    image_id = int(img_path.split('\\')[-1].split('_')[-1].split('.')[0])
    modality = img_path.split('\\')[-3]
    return subject_id, image_id, modality


def extract_indexes(img_pair_paths):
    indexes = [extract_index(img_path) for img_path in img_pair_paths]
    return indexes
