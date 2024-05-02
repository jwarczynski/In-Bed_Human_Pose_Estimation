import numpy as np

from src.evaluation.metrics import object_key_point_similarity
from src.evaluation.helper import reorder_skeleton

from src.inference import infere


def infere_and_compute_oks(model, image, cfg, gt_skeleton):
    gt_skeleton = [(gt[0].cpu().numpy()[0], gt[1].cpu().numpy()[0]) for gt in gt_skeleton]
    predicted_skeleton = infere_and_reorder_joints(model, image, cfg)

    return object_key_point_similarity(predicted_skeleton, gt_skeleton, image.shape)


def infere_and_reorder_joints(model, image, cfg):
    image_cpu = image.cpu().numpy()
    skeleton = infere(model, image_cpu, cfg)
    reordered_skeleton = reorder_skeleton(skeleton)
    predicted_skeleton = np.array([[x, y] for x, y, _ in reordered_skeleton])
    return predicted_skeleton
