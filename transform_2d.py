from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld, \
    RandRotate90d, RandFlipd, RandShiftIntensityd, RandCoarseDropoutd, RandGaussianNoised, RandWeightedCropd, \
    CropForegroundd

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"]),
        # RandWeightedCropd(
        #             keys=["image", "label"],
        #             spatial_size=[512, 512],
        #             num_samples=8,
        #             w_key="label",
        #         ),
        # CropForegroundd(keys=["image", "label"], source_key="image", fg_indices_key="fg_indices"),
        # RandCropByPosNegLabeld(
        #     keys=["image", "label"], label_key="label", spatial_size=[512, 512], pos=1, neg=1, num_samples=32
        # ),
        # RandCoarseDropoutd(keys=["image"], holes=8, spatial_size=[16, 32, 32], prob=0.30),
        # RandGaussianNoised(keys=["image"], prob=0.50, mean=0.0, std=0.2),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.30),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.30),
        # RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50)
        # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 2])
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image", "label"])
    ]
)