import numpy as np
import torch
import logging

from src.dataset.human_pose_dataset2 import HumanPoseDataset2

logger = logging.getLogger(__name__)


# noinspection PyMethodMayBeStatic
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, heatmap_generator, offset_generator=None, transforms=None):
        super().__init__()
        self.root = cfg.DATASET.ROOT
        self.dataset = dataset
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints_with_center = self.num_joints + 1

        self.sigma = cfg.DATASET.SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator
        self.transforms = transforms

        self.pose_dataset = HumanPoseDataset2(
            root_dir=self.root, modalities=('IR',),
            splits=cfg.DATASET.SPLITS, occlusions=cfg.DATASET.OCCLUSIONS,
            num_subjects=1, random_subjects=False,
            train=True, transform=None)

    def __len__(self):
        return self.pose_dataset.__len__()

    def __getitem__(self, idx):
        """
        :param idx (int): Index
        :return: tuple: (image, heatmap, mask, offset, offset_weight)
        """
        img, labels = self.pose_dataset[idx]
        mask = self.get_mask(img.IR.shape[0], img.IR.shape[1])
        joints, area = self.get_joints(labels)

        img, mask_list, joints_list, area = self.transforms(
            img.IR, [mask], [joints], area
        )

        heatmap, ignored = self.heatmap_generator(
            joints_list[0], self.sigma, self.center_sigma, self.bg_weight)
        mask = mask_list[0] * ignored

        offset, offset_weight = self.offset_generator(
            joints_list[0], area)

        return img, heatmap, mask, offset, offset_weight

    def get_mask(self, img_height, img_width):
        m = np.zeros((img_height, img_width))

        return m < 0.5

    # noinspection PyPep8Naming
    def get_joints(self, label):
        num_people = 1
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints_with_center, 3))

        gt_IR_torch = [l for l in label.IR]
        gt_skeleton = np.array([[gt[0], gt[1], 1] for gt in gt_IR_torch])

        joints[0, :self.num_joints, :3] = gt_skeleton

        area[0, 0] = self.cal_area_2_torch(
            torch.tensor(joints[0:1, :, :]))

        joints_sum = np.sum(joints[0, :-1, :2], axis=0)
        num_vis_joints = len(np.nonzero(joints[0, :-1, 2])[0])
        if num_vis_joints <= 0:
            joints[0, -1, :2] = 0
        else:
            joints[0, -1, :2] = joints_sum / num_vis_joints
        joints[0, -1, 2] = 1

        return joints, area

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h
