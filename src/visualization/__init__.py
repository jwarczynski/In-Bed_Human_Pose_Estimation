import matplotlib.pyplot as plt
from scipy.io import loadmat
import os


BODY_PARTS = [
    "Right ankle", "Right knee", "Right hip",
    "Left hip", "Left knee", "Left ankle",
    "Right wrist", "Right elbow", "Right shoulder",
    "Left shoulder", "Left elbow", "Left wrist",
    "Thorax", "Head top"
]


def show_img_annotated(axs, ir_img, rgb_img, gt=None):
    if type(ir_img) == str:
        # print(f"Reading {ir_img}")
        ir_img = plt.imread(ir_img)
    if type(rgb_img) == str:
        # print(f"Reading {rgb_img}")
        rgb_img = plt.imread(rgb_img)
    # ax.imshow(image, cmap='hot', extent=[0, image.shape[1], image.shape[0], 0])
    axs[0].imshow(ir_img, cmap='hot')
    axs[1].imshow(rgb_img)
    axs[0].axis('off')
    axs[1].axis('off')

    if gt is not None:
        annotate_img(axs[0], gt)
    return axs


def annotate_img(ax, gt):
    for i, body_part in enumerate(BODY_PARTS):
        if body_part.startswith('Right'):
            text_offset = (-35, 10)
        elif body_part.startswith('Left'):
            text_offset = (30, 10)
        else:
            text_offset = (30, 10)
        if body_part.split()[-1] == 'hip':
            text_offset = (text_offset[0], text_offset[1] - 20)
        ax.scatter(gt[0, i], gt[1, i], label=body_part)
        ax.annotate(body_part, (gt[0, i], gt[1, i]), textcoords="offset points", xytext=text_offset,
                    ha='center',
                    fontsize=8, color='white',
                    arrowprops=dict(facecolor='black', edgecolor='white', arrowstyle='->'))


def show_subject(subject_id='00001', num_images=3, root_dir='../res/dataset/train/train/'):
    gt_mat = None
    if os.path.exists(f'{root_dir}{subject_id}/joints_gt_IR.mat'):
        gt_file = f'{root_dir}{subject_id}/joints_gt_IR.mat'
        mat = loadmat(gt_file)
        gt_mat = mat['joints_gt']

    img_cover_type = os.listdir(f'{root_dir}{subject_id}/IR')[0]

    for i in range(num_images):
        fig, axs = plt.subplots(1, 2, figsize=(7, 5))
        ir_file = f"{root_dir}{subject_id}/IR/{img_cover_type}/image_{i + 1:06}.png"
        rgb_file = f'{root_dir}{subject_id}/RGB/{img_cover_type}/image_{i + 1:06}.png'
        if gt_mat is not None:
            gt = gt_mat[:, :, i]
        else:
            gt = None

        show_img_annotated(axs, ir_file, rgb_file, gt)
        plt.tight_layout()
        plt.show()