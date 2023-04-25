import os
from glob import glob
import cv2
import monai
import torch
from matplotlib import pyplot as plt
from monai.data import list_data_collate, DataLoader
from tqdm import tqdm
from transform_2d import train_transforms, val_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_2d_loader():
    train_files = []
    images_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image_512/image_1*.nii.gz"))
    label_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label_512/label_1*.nii.gz"))
    images_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image_512/image_2*.nii.gz"))
    label_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label_512/label_2*.nii.gz"))
    # images_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image_512/image_3*.nii.gz"))
    # label_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label_512/label_3*.nii.gz"))

    train1_files = [{"image": img, "label": seg} for img, seg in zip(images_1[:], label_1[:])]
    # val1_files = [{"image": img, "label": seg} for img, seg in zip(images_1[70:], label_1[70:])]
    train2_files = [{"image": img, "label": seg} for img, seg in zip(images_2[:], label_2[:])]
    # val2_files = [{"image": img, "label": seg} for img, seg in zip(images_2[290:], label_2[290:])]
    # train3_files = [{"image": img, "label": seg} for img, seg in zip(images_3[:50], label_3[:50])]
    # val3_files = [{"image": img, "label": seg} for img, seg in zip(images_3[:], label_3[:])]

    train_files.extend(train1_files)
    train_files.extend(train2_files)
    valid_files = [{"image": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/2d_volume_image/2d_surface_image_3"
                             ".nii.gz",
                    "label": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/2d_volume_label/2d_surface_label_3"
                             ".nii.gz"}]
    # train_files.extend(train3_files)
    # valid_files.extend(val1_files)
    # valid_files.extend(val2_files)
    # valid_files.extend(val3_files)

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    valid_ds = monai.data.Dataset(data=valid_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    return train_loader, valid_loader

# train_loader, valid_loader = get_loader()
#
# if not os.path.exists("save_dir_valid"):
#     os.makedirs("save_dir_valid")
# count = 0
# for batch_data in valid_loader:
#     img, label = batch_data["image"].to(device), batch_data["label"].to(device)
#
#     print(img.shape, label.shape)
#     for i in range(img.shape[0]):
#         img_ = img[i].cpu().numpy()
#         label_ = label[i].cpu().numpy()
#         plt.imshow(img_[img_.shape[0] // 2, :, :], cmap="gray")
#         plt.savefig(f"save_dir_valid/{count}_{i}_img.png")  # 保存图片
#         plt.clf()  # 清空当前 figure
#         plt.imshow(label_[0, :, :])
#         plt.savefig(f"save_dir_valid/{count}_{i}_label.png")  # 保存图片
#         plt.clf()  # 清空当前 figure
#     count += 1
#         # plt.show()
