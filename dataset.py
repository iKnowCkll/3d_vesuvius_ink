from glob import glob

import cv2
import monai
import torch
from matplotlib import pyplot as plt

from monai.data import list_data_collate, DataLoader
from tqdm import tqdm

from transforms import train_transforms, val_transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def get_loader():
#     train_files = []
#     valid_files = []
#     images_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_1*.nii.gz"))
#     label_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_1*.nii.gz"))
#     images_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_2*.nii.gz"))
#     label_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_2*.nii.gz"))
#     images_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_3*.nii.gz"))
#     label_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_3*.nii.gz"))
#
#     train1_files = [{"image": img, "label": seg} for img, seg in zip(images_1[:25], label_1[:25])]
#     val1_files = [{"image": img, "label": seg} for img, seg in zip(images_1[25:], label_1[25:])]
#     train2_files = [{"image": img, "label": seg} for img, seg in zip(images_2[:70], label_2[:70])]
#     val2_files = [{"image": img, "label": seg} for img, seg in zip(images_2[70:], label_2[70:])]
#     train3_files = [{"image": img, "label": seg} for img, seg in zip(images_3[:25], label_3[:25])]
#     val3_files = [{"image": img, "label": seg} for img, seg in zip(images_3[25:], label_3[25:])]
#
#     train_files.extend(train1_files)
#     train_files.extend(train2_files)
#     train_files.extend(train3_files)
#     valid_files.extend(val1_files)
#     valid_files.extend(val2_files)
#     valid_files.extend(val3_files)
#
#     train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
#     valid_ds = monai.data.Dataset(data=valid_files, transform=val_transforms)
#     train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=list_data_collate,
#                               pin_memory=torch.cuda.is_available())
#     valid_loader = DataLoader(valid_ds, batch_size=8, num_workers=4, collate_fn=list_data_collate)
#     return train_loader, valid_loader


def get_volume_loader():
    # train_files = [{"image": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_image/surface_volume1.nii
    # .gz", "label": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_label/label_volume1.nii.gz"},
    # {"image": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_image/surface_volume2.nii.gz",
    # "label": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_label/label_volume2.nii.gz"}]
    train_files = []
    images_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_1*.nii.gz"))
    label_1 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_1*.nii.gz"))
    images_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_2*.nii.gz"))
    label_2 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_2*.nii.gz"))
    images_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/image/image_3*.nii.gz"))
    label_3 = sorted(glob("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/label/label_3*.nii.gz"))

    train1_files = [{"image": img, "label": seg} for img, seg in zip(images_1[:], label_1[:])]
    train2_files = [{"image": img, "label": seg} for img, seg in zip(images_2[:], label_2[:])]
    valid_files = [{"image": img, "label": seg} for img, seg in zip(images_3[:], label_3[:])]
    train_files.extend(train1_files)
    train_files.extend(train2_files)

    # valid_files = [{"image": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_image/surface_volume3.nii
    # .gz", "label": "/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/volume_label/label_volume3.nii.gz"}]
    valid_mask = cv2.imread("/home/chenk/model_train/kaggle/3d_vesuvius_ink/data/mask3.png", 0)
    valid_mask = valid_mask.astype("float32")
    valid_mask = valid_mask / 255.0

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    valid_ds = monai.data.Dataset(data=valid_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=list_data_collate,
                              pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(valid_ds, batch_size=16, num_workers=4, collate_fn=list_data_collate)
    return train_loader, valid_loader, valid_mask


# train_loader, valid_loader = get_volume_loader()
# for batch_data in valid_loader:
#     img, label = batch_data["image"].to(device), batch_data["label"].to(device)
#
#     print(img.shape, label.shape)
#     img = img[0][0].cpu().numpy()
#     label = label[0][0].cpu().numpy()
#     plt.imshow(img[img.shape[0] // 2, :, :], cmap="gray")
#     plt.show()
#     plt.imshow(label[label.shape[0] // 2, :, :])
#     plt.show()
