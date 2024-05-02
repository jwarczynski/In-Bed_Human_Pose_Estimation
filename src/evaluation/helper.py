from src.visualization import BODY_PARTS

import numpy as np

MODEL_JOINTS_ORDER = [
    "Left shoulder", "Right shoulder", 'Left elbow',
    'Right elbow', 'Left wrist', 'Right wrist',
    "Left hip", "Right hip", "Left knee",
    "Right knee", "Left ankle", "Right ankle",
    "Head top", "Thorax"
]
# Generate a dictionary to map joint names to indices
JOINT_INDICES = {joint: index for index, joint in enumerate(MODEL_JOINTS_ORDER)}


def reorder_skeleton(skeleton):
    # Reorder the skeleton array
    reordered_skeleton = np.zeros_like(skeleton)
    for joint_name, index in JOINT_INDICES.items():
        joint_index_in_body_parts = BODY_PARTS.index(joint_name)
        reordered_skeleton[joint_index_in_body_parts] = skeleton[index]
    return reordered_skeleton
