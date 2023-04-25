import monai
import numpy as np
from PIL import Image
import nibabel as nib
from matplotlib import pyplot as plt
from monai.data import DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd
import os
# # #
# dir_names = [d for d in os.listdir('E:/kaggle/3d_vesuvius_ink_detection/2048_data/3')]
#
# print(dir_names)
#
# for i in dir_names:
#     # k, v = os.path.splitext(i)
#     im_list = []
#     for j in range(28, 36):
#         img = Image.open(f'E:/kaggle/3d_vesuvius_ink_detection/2048_data/3/{i}/{j}.png')
#         im_arr = np.array(img)
#         im_list.append(im_arr)
#     im = np.stack(im_list, axis=0)
#
#     # 为图像添加一个通道数
#     # im = np.expand_dims(im, axis=0)
#     print(im.shape)
#     # 创建Nifti格式的图像数据
#     nii_img = nib.Nifti1Image(im, np.eye(4))
#     #
#     # 保存Nifti格式的图像数据到文件
#     nib.save(nii_img, f'E:/kaggle/3d_vesuvius_ink_detection/2048_data/3/image3_{i}.nii.gz')

# import os
#
# dir_names = [d for d in os.listdir('E:/kaggle/3d_vesuvius_ink_detection/512_data/3_label')]
# print(dir_names)
#
# for i in dir_names:
#     img = Image.open(f'E:/kaggle/3d_vesuvius_ink_detection/512_data/3_label/{i}')
#     k, v = os.path.splitext(i)
#     # im_list = []
#     # for j in range(28, 36):
#     #     img = Image.open(f'E:/kaggle/3d_vesuvius_ink_detection/512_data/3/{i}/{j}.png')
#     #     im_arr = np.array(img)
#     #     im_list.append(im_arr)
#     # img = np.stack(im_list, axis=0)
#
#     img = np.expand_dims(img, axis=0)
#     print(img.shape)
#     # 创建Nifti格式的图像数据
#     nii_img = nib.Nifti1Image(img, np.eye(4))
#     #
#     # 保存Nifti格式的图像数据到文件
#     nib.save(nii_img, f'E:/kaggle/3d_vesuvius_ink_detection/512_data/3_label/label_3_{k}.nii.gz')
# #
# nib.save(nii_img, 'E:/kaggle/btcv/stacked_label.nii.gz')

# train_files = [{"image": "E:/kaggle/btcv/stacked_image.nii.gz", "label": "E:/kaggle/btcv/stacked_label.nii.gz"}]
# val_transforms = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys=["image", "label"]),
#         ScaleIntensityd(keys=["image", "label"]),
#     ]
# )
#
# check_ds = monai.data.Dataset(data=train_files, transform=val_transforms)
# check_loader = DataLoader(check_ds, batch_size=1, num_workers=0)
# check_data = monai.utils.misc.first(check_loader)
# img1 = check_data["image"][0][0].numpy()
# seg1 = check_data["label"][0][0].numpy()
# plt.imshow(img1[:, :, img1.shape[2] // 2], cmap="gray")
# plt.show()
# plt.imshow(seg1[:, :, seg1.shape[2] // 2])
# plt.show()
#
#
# im_list = []
# for j in range(28, 36):
#     img = Image.open(f'E:/kaggle/vesuvius-ink-detection/train/3/surface_volume/{j}.png')
#     im_arr = np.array(img)
#     im_list.append(im_arr)
# im = np.stack(im_list, axis=0)
#
# # 为图像添加一个通道数
# # im = np.expand_dims(im, axis=0)
# print(im.shape)
# # 创建Nifti格式的图像数据
# nii_img = nib.Nifti1Image(im, np.eye(4))
# #
# # 保存Nifti格式的图像数据到文件
# nib.save(nii_img, 'E:/kaggle/vesuvius-ink-detection/train/3/2d_surface_image_3.nii.gz')

im_list = []
# for j in range(28, 36):
img = Image.open(f'E:/kaggle/vesuvius-ink-detection/train/3/inklabels.png')
im_arr = np.array(img)
# im_list.append(im_arr)
# im = np.stack(im_list, axis=0)

# 为图像添加一个通道数
im = np.expand_dims(im_arr, axis=0)
print(im.shape)
# 创建Nifti格式的图像数据
nii_img = nib.Nifti1Image(im, np.eye(4))
#
# 保存Nifti格式的图像数据到文件
nib.save(nii_img, 'E:/kaggle/vesuvius-ink-detection/train/3/2d_surface_label_3.nii.gz')