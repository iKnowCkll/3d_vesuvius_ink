import gc
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

from metric import calc_fbeta
from utils.other_utils import AverageMeter


def train_fn(epoch, train_loader, model, criterion, optimizer, device, writer):
    model.train()

    scaler = GradScaler(enabled=True)
    train_loss = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train epoch:{epoch}')
    for i, batch_data in pbar:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        batch_size = inputs.size()[0]
        optimizer.zero_grad()

        with autocast(enabled=True):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        train_loss.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{train_loss.avg:0.4f}',
                         current_lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    writer.add_scalar("train_loss", train_loss.avg, epoch)
    torch.cuda.empty_cache()
    gc.collect()

    return train_loss.avg


def valid_fn(epoch, valid_loader, valid_mask, model, criterion, device, post_trans, writer):
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    valid_loss = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid epoch:{epoch}')
    with torch.no_grad():
        for i, val_data in pbar:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            batch_size = val_images.size()[0]
            roi_size = (16, 512, 512)
            sw_batch_size = 4

            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            loss = criterion(val_outputs, val_labels)  # val_outputs: (1, 1, 16, 1024, 1024)
            valid_loss.update(loss.item(), batch_size)

            val_outputs = post_trans(val_outputs)  # val_outputs: (1, 1, 16, 1024, 1024)
            dice_metric(y_pred=val_outputs, y=val_labels)
            # metric = calc_fbeta(val_labels, val_outputs, valid_mask)

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(valid_loss=f'{valid_loss.avg:0.4f}',
                             gpu_memory=f'{mem:0.2f} GB')
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar("valid_dice", metric, epoch)
        writer.add_scalar("valid_loss", valid_loss.avg, epoch)

        plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
        plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")

        torch.cuda.empty_cache()
        gc.collect()

    return valid_loss.avg, metric


def train_2d_fn(epoch, train_loader, model, criterion, optimizer, device, writer):
    model.train()

    scaler = GradScaler(enabled=True)
    train_loss = AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train epoch:{epoch}')
    for i, batch_data in pbar:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        batch_size = inputs.size()[0]
        optimizer.zero_grad()

        with autocast(enabled=True):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        train_loss.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{train_loss.avg:0.4f}',
                         current_lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    writer.add_scalar("train_loss", train_loss.avg, epoch)
    torch.cuda.empty_cache()
    gc.collect()

    return train_loss.avg


def valid_2d_fn(epoch, valid_loader, model, criterion, device, post_trans, writer):
    model.eval()

    valid_loss = AverageMeter()
    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid epoch:{epoch}')
    with torch.no_grad():
        for i, val_data in pbar:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
            batch_size = val_images.size()[0]
            roi_size = (512, 512)
            sw_batch_size = 16

            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            loss = criterion(val_outputs, val_labels)  # val_outputs: (1, 1, size, size)
            valid_loss.update(loss.item(), batch_size)

            val_outputs = post_trans(val_outputs)  # val_outputs: (1, 1, size, size)

            metric = calc_fbeta(val_labels, val_outputs)

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            pbar.set_postfix(valid_loss=f'{valid_loss.avg:0.4f}',
                             gpu_memory=f'{mem:0.2f} GB')

        writer.add_scalar("valid_dice", metric, epoch)
        writer.add_scalar("valid_loss", valid_loss.avg, epoch)

        plot_2d_or_3d_image(val_images, epoch, writer, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch, writer, index=0, tag="label")
        plot_2d_or_3d_image(val_outputs, epoch, writer, index=0, tag="output")

        torch.cuda.empty_cache()
        gc.collect()

    return valid_loss.avg, metric
