# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200  # print log every 20 steps
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets[0] = targets[0].to(device, non_blocking=True)
        targets[1] = targets[1].to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            AU_loss = criterion(outputs[0], targets[0])
            ID_loss = criterion(outputs[1], targets[1])
        loss_value = AU_loss.item() + ID_loss.item()
        loss = AU_loss + ID_loss

        # handle nan loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        # update model parameters
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        # log into tensorboard
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_100x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 100)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1][1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output[1], target)

        if len(output[1].shape) == len(target.shape):
            target = torch.argmax(target, dim=1)
        acc1, acc5 = accuracy(output[1], target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def AU_evaluate(data_loader, model, device):
    criterion = torch.nn.BCEWithLogitsLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()  # switch to evaluation mode
    all_preds = []
    all_targets = []

    for batch in metric_logger.log_every(data_loader, 200, header):
        images = batch[0]
        target = batch[-1][0]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)  # 2D tensor (batch, 12)

        # compute loss
        with torch.cuda.amp.autocast():
            output = model(images)  # 2D tensor (batch, 12)
            loss = criterion(output[0], target)
        metric_logger.update(loss=loss.item())

        # for f1 computation
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(output[0])
        all_preds.append(probs.detach().cpu().numpy())
        all_targets.append(target.detach().cpu().numpy())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    # compute f1
    y_probs = np.concatenate([arr for arr in all_preds], axis=0)
    y_true = np.concatenate([arr for arr in all_targets], axis=0)

    # AP = []
    # for cls in range(y_true.shape[1]):
    #     AP.append(average_precision_score(y_true[:, cls], y_probs[:, cls]))
    # # print(f"AP for each category: {AP}")
    # mAP = np.mean(AP)

    # each AU uses a different threshold
    # f1_score_ls = []
    # for i in range(1, 100):
    #     threshold = i * 0.01
    #     y_pred = np.zeros(y_probs.shape)
    #     y_pred[np.where(y_probs >= threshold)] = 1
    #
    #     # Compute F1 score for each class
    #     f1_scores = []
    #     for class_idx in range(y_true.shape[1]):
    #         f1_scores.append(f1_score(y_true[:, class_idx], y_pred[:, class_idx]))
    #     f1_score_ls.append(f1_scores)
    #
    #     # compute accuracy
    #     roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    #     accuracy = accuracy_score(y_true, y_pred)
    #
    # f1_score_arr = np.array(f1_score_ls)
    # avg_best_f1 = np.max(f1_score_arr, axis=0)
    # print(f"f1_mean: {avg_best_f1.mean()}, best_f1_scores: {avg_best_f1}")

    # all AUs use 0.5 threshold
    threshold = 0.5
    y_pred = np.zeros(y_probs.shape)
    y_pred[np.where(y_probs >= threshold)] = 1

    # Compute F1 score for each class
    f1_scores = []
    for class_idx in range(y_true.shape[1]):
        f1_scores.append(f1_score(y_true[:, class_idx], y_pred[:, class_idx]))
    f1_score_arr = np.array(f1_scores)
    print(f"f1_mean: {f1_score_arr.mean()} with threshold 0.5, f1_scores: {f1_score_arr}")

    # all AUs use the same threshold
    f1_score_ls = []
    for i in range(1, 100):
        threshold = i * 0.01
        y_pred = np.zeros(y_probs.shape)
        y_pred[np.where(y_probs >= threshold)] = 1

        # Compute F1 score for each class
        f1_scores = []
        for class_idx in range(y_true.shape[1]):
            f1_scores.append(f1_score(y_true[:, class_idx], y_pred[:, class_idx]))
        f1_score_ls.append(f1_scores)

    f1_score_arr = np.array(f1_score_ls)
    max_f1_row_index = np.argmax(np.mean(f1_score_arr, axis=1))
    max_mean_row = f1_score_arr[max_f1_row_index]
    print(f"f1_mean: {max_mean_row.mean()} with threshold {(max_f1_row_index+1)/100}, best_f1_scores: {max_mean_row}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, max_mean_row.mean()