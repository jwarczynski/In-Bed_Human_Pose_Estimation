from easydict import EasyDict
import matplotlib.pyplot as plt
from cv2 import imread
from scipy.io import loadmat
import pandas as pd
import numpy as np
import glob
import os

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from src.visualization import BODY_PARTS

PATH_SEP = os.path.sep


class HumanPoseDataset2(Dataset):
    def __init__(self, root_dir, modalities=('IR', 'RGB'),
                 splits=('train, test1, test2', 'augmented', 'valid'),
                 occlusions=('uncover', 'cover1', 'cover2'),
                 num_subjects=None, random_subjects=False,
                 positions=None, train=False, transform=None):
        self._root_dir = root_dir
        self._transform = transform
        self._train = train
        self._modalities = modalities
        self._splits = [split if split != 'augmented' else 'train2' for split in splits]
        self._occlusions = occlusions
        self._num_subjects = num_subjects
        self._random_subjects = random_subjects
        self._positions = positions
        self._img_paths = np.array(self._get_img_paths())
        if self._train:
            self._ground_truth_df = self._get_ground_truth()

    def _get_img_paths(self):
        img_paths = []
        for split in self._splits:
            for subject_dir in glob.glob(f'{self._root_dir}/{split}/{split}/*'):
                if self._train and glob.glob(f'{subject_dir}/*.mat') == []:
                    # if train skip subjects without ground truth data
                    continue
                present_modality_dirs = glob.glob(f'{subject_dir}/*/')
                modality_dirs = [modality_dir for modality_dir in present_modality_dirs
                                 if modality_dir.split(PATH_SEP)[-2] in self._modalities]
                # modality_img_pairs = zip(*[glob.glob(f'{modality}/*/*.png') for modality in modality_dirs])
                # Construct the list of glob patterns based on occlusions
                modality_img_pairs = []
                for modality in modality_dirs:
                    for occlusion in self._occlusions:
                        modality_img_pairs.extend(glob.glob(f'{modality}/{occlusion}/*.png'))
                img_paths.extend(modality_img_pairs)
        occlusion_img_paths = []
        for path in img_paths:
            if any(occlusion in str(path) for occlusion in self._occlusions):
                occlusion_img_paths.append(path)
        return occlusion_img_paths

    def _get_ground_truth(self):
        col_names = ['Subject_id', 'Image_id', 'Modality', 'Split',]
        col_names.extend(BODY_PARTS)
        gt_df = pd.DataFrame(columns=col_names)

        splits = set(self._splits).intersection(['train', 'train2', 'valid'])
        for split in splits:
            split_dir = os.path.join(self._root_dir, split, split)
            for subject_dir in os.listdir(str(split_dir)):
                gt_matrices = glob.glob(f'{os.path.join(str(split_dir), subject_dir)}/*.mat')
                modalities = [gt_file.split('.mat')[0].split('_')[-1] for gt_file in gt_matrices]
                subject_gt = get_ground_truth_for_subject(gt_matrices, modalities, subject_dir, split)
                gt_df = pd.concat([gt_df, pd.DataFrame(subject_gt, columns=col_names)], ignore_index=True)

        gt_df.set_index(['Subject_id', 'Image_id', 'Modality', 'Split'], inplace=True)
        return gt_df

    def __len__(self):
        return len(self._img_paths)
    
    def __getitem__(self, idx):
        img_pair_paths = self._img_paths[idx]
        indexes = extract_indexes(img_pair_paths)
        images = [imread(img_path) for img_path in [img_pair_paths]]
    
        if self._transform:
            images = (self._transform(img) for img in images)
    
        images = EasyDict({path.split(PATH_SEP)[-3]: img for path, img in zip([img_pair_paths], images)})
        labels = EasyDict()
        if self._train:
            labels = EasyDict({idx[-2]: self._ground_truth_df.loc[idx].values.tolist() for idx in [indexes]})

        for modal in self._modalities:
            if modal not in labels:
                labels[modal] = []
    
        return images, labels
    

def get_ground_truth_for_subject(gt_matrices, modalities, subject_dir, split):
    subject_gt = []
    for gt_matrix, modality in zip(gt_matrices, modalities):
        gt = loadmat(gt_matrix)
        for i in range(gt['joints_gt'].shape[2]):
            x_parts_locations = tuple(gt['joints_gt'][0, :, i])
            y_parts_locations = tuple(gt['joints_gt'][1, :, i])
            if_part_occluded = tuple(gt['joints_gt'][2, :, i])
            parts_gt = tuple(zip(x_parts_locations, y_parts_locations, if_part_occluded))

            row = [int(subject_dir), i + 1, modality, split]
            row.extend(parts_gt)

            subject_gt.append(row)

    return subject_gt


def extract_indexes(img_pair_paths):
    if not isinstance(img_pair_paths, list):
        return extract_index(img_pair_paths)
    indexes = [extract_index(img_path) for img_path in img_pair_paths]
    return indexes


def extract_index(img_path):
    subject_id = int(img_path.split(PATH_SEP)[-4])
    image_id = int(img_path.split(PATH_SEP)[-1].split('_')[-1].split('.')[0])
    modality = img_path.split(PATH_SEP)[-3]
    split = img_path.split(PATH_SEP)[-5]
    return subject_id, image_id, modality, split.split('/')[-1]
