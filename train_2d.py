import gc
import os
import time
import warnings

import monai
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset_2d import get_2d_loader
from model import build_2d_model
from utils.other_utils import set_seed
from colorama import Fore, Style

from utils.train_utils import train_2d_fn, valid_2d_fn

g_ = Fore.GREEN
r_ = Fore.RED
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
set_seed()
num_epochs = 500

save_path = "/home/chenk/model_train/kaggle/3d_vesuvius_ink/save_2d_model/100_valid/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = SummaryWriter("/home/chenk/model_train/kaggle/3d_vesuvius_ink/save_2d_model/100_valid/runs")

train_loader, valid_loader = get_2d_loader()
model = build_2d_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
loss_function = monai.losses.DiceLoss(sigmoid=True)
post_trans = monai.transforms.Compose([monai.transforms.Activations(sigmoid=True)])


def run_training():
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))
    best_metric_epoch = -1
    best_metric = -1
    metric = -1
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print("-" * 10)
        print(f"epoch {epoch}/{num_epochs}")
        epoch_start_time = time.time()
        train_loss = train_2d_fn(epoch=epoch, train_loader=train_loader, model=model,
                                 criterion=loss_function, optimizer=optimizer, device=device, writer=writer)
        if epoch % 100 == 0:
            valid_loss, metric = valid_2d_fn(epoch=epoch, valid_loader=valid_loader, model=model,
                                          criterion=loss_function, device=device,
                                          post_trans=post_trans, writer=writer)
        scheduler.step(epoch)
        elapsed = time.time() - epoch_start_time
        print(f"epoch {epoch} train loss: {train_loss:.4f} time: {elapsed:.0f}s")
        if metric > best_metric:
            best_metric = metric
            best_metric_epoch = epoch
            torch.save(model.state_dict(), save_path + "best_metric_model.pth")
            print(f'{g_}Save Best Score: {best_metric:.4f} at epoch: {best_metric_epoch}{sr_}')
    total_time = time.time() - start_time
    print(f"{r_}train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
          f" total time elapsed: {total_time // 3600:.0f}h "
          f"{total_time % 3600 // 60:.0f}m {(total_time % 3600) % 60:.0f}s{sr_}")
    writer.close()


if __name__ == "__main__":
    run_training()
