import monai
import torch
from monai.networks.nets import UNETR

from dataset_2d import get_2d_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# def build_model():
#     model = monai.networks.nets.UNet(
#         spatial_dims=3,
#         in_channels=1,
#         out_channels=1,
#         channels=(16, 32, 64, 128, 256),
#         strides=(2, 2, 2, 2),
#         num_res_units=2,
#     ).to(device)
#     return model


def build_2d_model():
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=8,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    return model


train_loader, valid_loader = get_2d_loader()

# model = build_2d_model()
# with torch.no_grad():
#     check_data = next(iter(train_loader))
#     image = check_data["image"].to(device)
#     label = check_data["label"].to(device)
#     print(image.shape, label.shape)
#     print(model(image).shape)

# model = UNETR(
#     in_channels=1,
#     out_channels=1,
#     img_size=(16, 1024, 1024),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     pos_embed="perceptron",
#     norm_name="instance",
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)
# model.eval()
