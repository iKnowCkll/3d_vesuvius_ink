import cv2
import numpy as np
import torch
from torch import tensor


# output = torch.randn(1, 1, 16, 256, 256)
# output = output.squeeze(dim=1).mean(dim=0)  # 消除维度为1，求平均
# output = output.cpu().numpy()  # 转换为NumPy数组
# output = output[0]  # 得到（256，256）形状的数组
# print(output.shape)
#
# output = tensor([[[[[1, 0., 1., 1.],
#                     [1, 1., 0., 1.],
#                     [0, 1., 1., 1.],
#                     [1., 0., 0., 0.]],
#
#                    [[0.5, 1., 1., 1.],
#                     [0.1, 0., 0., 1.],
#                     [0., 0., 1., 1.],
#                     [0., 1., 0., 0.]]]]])
# output = output.squeeze().mean(dim=0, dtype=torch.float32)  # 消除维度为1，求平均
#
# output = output.cpu().numpy()
# print(output.dtype) # 转换为NumPy数组
# valid_mask = np.zeros((4, 4))
# output = valid_mask * output
# print(output)

#
def calc_3d_fbeta(targets, preds, mask, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    # targets = targets.squeeze().mean(dim=0, dtype=torch.float32)  # (1, 1, 16, size, size) -> (size, size)
    # preds = preds.squeeze().mean(dim=0, dtype=torch.float32)  # (1, 1, 16, size, size) -> (size, size)
    # targets = targets.cpu().numpy()
    # preds = preds.cpu().numpy()
    # preds = preds * mask

    targets = targets.astype(int).flatten()
    preds = preds.flatten()
    preds = (preds >= 0.1).astype(int)

    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


# def calc_fbeta(mask, mask_pred):
#     mask = mask.astype(int).flatten()
#     mask_pred = mask_pred.flatten()
#
#     best_th = 0
#     best_dice = 0
#     for th in np.array(range(10, 75 + 1, 5)) / 100:
#
#         # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
#         dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
#
#         if dice > best_dice:
#             best_dice = dice
#             best_th = th
#
#     return best_dice, best_th

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    dice = fbeta_numpy(mask, (mask_pred >= 0.4).astype(int), beta=0.5)

    return dice

#
# def calc_cv(mask_gt, mask_pred):
#
#     # dice = calc_fbeta(mask_gt, mask_pred)
#     best_dice, best_th = calc_fbeta(mask_gt, mask_pred)
#
#     return best_dice, best_th
