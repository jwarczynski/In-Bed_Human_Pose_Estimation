import torch
import torchvision

import numpy as np

from src.lib.core.inference import get_multi_stage_outputs
from src.lib.core.inference import aggregate_results
from src.lib.core.nms import pose_nms

from src.lib.utils.transforms import resize_align_multi_scale
from src.lib.utils.transforms import get_final_preds
from src.lib.utils.transforms import get_multi_scale_size


def infere(model, image, cfg):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
        )

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            # Resize the image
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )
            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).cuda()

            # Get multi-stage outputs from the model
            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )

            # Aggregate results
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )

        # Average the heatmaps
        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)

        # Perform pose non-maximum suppression
        poses, scores = pose_nms(cfg, heatmap_avg, poses)
        pose = np.argmax(scores)

        # Get final predictions
        final_poses = get_final_preds(
            poses, center, scale_resized, base_size
        )

    return final_poses[pose]
