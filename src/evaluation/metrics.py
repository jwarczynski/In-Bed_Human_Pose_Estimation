import numpy as np


def object_key_point_similarity(skeleton1, skeleton2, img_shape):
    '''
    :param skeleton1: list of tuples (x, y) of keypoints
    :param skeleton2: list of tuples (x, y) of keypoints
    :param img_shape: tuple (height, width, optional channels) of the image
    :return: OKS metric defined in https://cocodataset.org/#keypoints-eval
    '''

    '''
     constants defined is same order as BODY_PARTS
     based on https://stasiuk.medium.com/pose-estimation-metrics-844c07ba0a78
    '''
    keypoint_constants = [
        0.089, 0.087, 0.107,
        0.107, 0.087, 0.087,
        0.062, 0.072, 0.079,
        0.079, 0.072, 0.062,
        0.079, 0.079
    ]

    similarity = 0
    scale_squred = img_shape[0] ** 2 + img_shape[1] ** 2
    for joint1, joint2, keypoint_const in zip(skeleton1, skeleton2, keypoint_constants):
        x1, y1 = joint1
        x2, y2 = joint2
        euclidean_dist = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        similarity += np.exp(-euclidean_dist ** 2 / (2 * scale_squred * keypoint_const ** 2))
    return similarity / len(skeleton1)


def euclidean_distance(skeleton1, skeleton2):
    return np.linalg.norm(skeleton1 - skeleton2, axis=1).mean()
