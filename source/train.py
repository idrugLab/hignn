# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-10-01
@File    : train.py
"""

import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from utils import create_logger, seed_set
from utils import NoamLR, build_scheduler, build_optimizer, get_metric_func
from utils import load_checkpoint, save_best_checkpoint, load_best_result
from dataset import build_loader
from loss import bulid_loss
from model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="codes for HiGNN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="../configs/bbbp.yaml",
        type=str,
    )

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for training")
    parser.add_argument('--lr_scheduler', type=str, help='learning rate scheduler')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

    args = parser.parse_args()
    cfg = get_config(args)

    return args, cfg


def train_one_epoch(cfg, model, criterion, trainloader, optimizer, lr_scheduler, device, logger):
    model.train()

    losses = []
    y_pred_list = {}
    y_label_list = {}

    for data in trainloader:
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output, vec1, vec2 = output
        else:
            output, vec1, vec2 = output, None, None
        loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]

                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                loss += criterion[i](y_pred, y_label, vec1, vec2)
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                loss += criterion(y_pred, y_label, vec1, vec2)
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step()

        losses.append(loss.item())

    # Compute metric
    results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while training')

            if nan:
                results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_results = np.nanmean(results)
    trn_loss = np.array(losses).mean()

    return trn_loss, avg_results


@torch.no_grad()
def validate(cfg, model, criterion, dataloader, epoch, device, logger, eval_mode=False):
    model.eval()

    losses = []
    y_pred_list = {}
    y_label_list = {}

    for data in dataloader:
        data = data.to(device)
        output = model(data)
        if isinstance(output, tuple):
            output, vec1, vec2 = output
        else:
            output, vec1, vec2 = output, None, None
        loss = 0

        for i in range(len(cfg.DATA.TASK_NAME)):
            if cfg.DATA.TASK_TYPE == 'classification':
                y_pred = output[:, i * 2:(i + 1) * 2]
                y_label = data.y[:, i].squeeze()
                validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
                if len(validId) == 0:
                    continue
                if y_label.dim() == 0:
                    y_label = y_label.unsqueeze(0)

                y_pred = y_pred[torch.tensor(validId).to(device)]
                y_label = y_label[torch.tensor(validId).to(device)]

                loss += criterion[i](y_pred, y_label, vec1, vec2)
                y_pred = F.softmax(y_pred.detach().cpu(), dim=-1)[:, 1].view(-1).numpy()
            else:
                y_pred = output[:, i]
                y_label = data.y[:, i]
                loss += criterion(y_pred, y_label, vec1, vec2)
                y_pred = y_pred.detach().cpu().numpy()

            try:
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            except:
                y_label_list[i] = []
                y_pred_list[i] = []
                y_label_list[i].extend(y_label.cpu().numpy())
                y_pred_list[i].extend(y_pred)
            losses.append(loss.item())

    # Compute metric
    val_results = []
    metric_func = get_metric_func(metric=cfg.DATA.METRIC)
    for i, task in enumerate(cfg.DATA.TASK_NAME):
        if cfg.DATA.TASK_TYPE == 'classification':
            nan = False
            if all(target == 0 for target in y_label_list[i]) or all(target == 1 for target in y_label_list[i]):
                nan = True
                logger.info(f'Warning: Found task "{task}" with targets all 0s or all 1s while validating')

            if nan:
                val_results.append(float('nan'))
                continue

        if len(y_label_list[i]) == 0:
            continue

        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results)
    val_loss = np.array(losses).mean()
    if eval_mode:
        logger.info(f'Seed {cfg.SEED} Dataset {cfg.DATA.DATASET} ==> '
                    f'The best epoch:{epoch} test_loss:{val_loss:.3f} test_scores:{avg_val_results:.3f}')
        return val_results

    return val_loss, avg_val_results


def train(cfg, logger):
    seed_set(cfg.SEED)
    # step 1: dataloder loading, get number of tokens
    train_loader, val_loader, test_loader, weights = build_loader(cfg, logger)
    # step 2: model loading
    model = build_model(cfg)
    logger.info(model)
    # device mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # step 3: optimizer loading
    optimizer = build_optimizer(cfg, model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    # step 4: lr_scheduler loading
    lr_scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    # step 5: loss function loading
    if weights is not None:
        criterion = [bulid_loss(cfg, torch.Tensor(w).to(device)) for w in weights]
    else:
        criterion = bulid_loss(cfg)

    # step 6: tensorboard loading
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        tensorboard_dir = os.path.join(cfg.OUTPUT_DIR, "tensorboard")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
    else:
        tensorboard_dir = None

    if tensorboard_dir is not None:
        writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        writer = None

    # step 7: model resuming (if training is interrupted, this will work.)
    best_epoch, best_score = 0, 0 if cfg.DATA.TASK_TYPE == 'classification' else float('inf')
    if cfg.TRAIN.RESUME:
        best_epoch, best_score = load_checkpoint(cfg, model, optimizer, lr_scheduler, logger)
        validate(cfg, model, criterion, val_loader, best_epoch, device, logger)

        if cfg.EVAL_MODE:
            return

    # step 8: training loop
    logger.info("Start training")
    early_stop_cnt = 0
    start_time = time.time()
    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.MAX_EPOCHS):

        # 1: Results after one epoch training
        trn_loss, trn_score = train_one_epoch(cfg, model, criterion, train_loader, optimizer,
                                              lr_scheduler, device, logger)
        val_loss, val_score = validate(cfg, model, criterion, val_loader, epoch, device, logger)
        # Just for observing the testset results during training
        test_loss, test_score = validate(cfg, model, criterion, test_loader, epoch, device, logger)

        # 2: Upadate learning rate
        if not isinstance(lr_scheduler, NoamLR):
            lr_scheduler.step(val_loss)

        # 3: Print results
        if epoch % cfg.SHOW_FREQ == 0 or epoch == cfg.TRAIN.MAX_EPOCHS - 1:
            lr_cur = lr_scheduler.optimizer.param_groups[0]['lr']
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} trn_loss:{trn_loss:.3f} '
                        f'trn_{cfg.DATA.METRIC}:{trn_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} val_loss:{val_loss:.3f} '
                        f'val_{cfg.DATA.METRIC}:{val_score:.3f} lr:{lr_cur:.5f}')
            logger.info(f'Epoch:{epoch} {cfg.DATA.DATASET} test_loss:{test_loss:.3f} '
                        f'test_{cfg.DATA.METRIC}:{test_score:.3f} lr:{lr_cur:.5f}')

        # 4: Tensorboard for training visualization.
        loss_dict, acc_dict = {"train_loss": trn_loss}, {f"train_{cfg.DATA.METRIC}": trn_score}
        loss_dict["valid_loss"], acc_dict[f"valid_{cfg.DATA.METRIC}"] = val_loss, val_score

        if cfg.TRAIN.TENSORBOARD.ENABLE:
            writer.add_scalars(f"scalar/{cfg.DATA.METRIC}", acc_dict, epoch)
            writer.add_scalars("scalar/loss", loss_dict, epoch)

        # 5: Save best results.
        if cfg.DATA.TASK_TYPE == 'classification' and val_score > best_score or \
                cfg.DATA.TASK_TYPE == 'regression' and val_score < best_score:
            best_score, best_epoch = val_score, epoch
            save_best_checkpoint(cfg, epoch, model, best_score, best_epoch, optimizer, lr_scheduler, logger)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        # 6: Early stopping.
        if early_stop_cnt > cfg.TRAIN.EARLY_STOP > 0:
            logger.info('Early stop hitted!')
            break

    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    # 7: Record training time.
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'Training time {total_time_str}')

    # 8: Evaluation.
    model, best_epoch = load_best_result(cfg, model, logger)
    score = validate(cfg, model, criterion, test_loader, best_epoch, device, logger=logger, eval_mode=True)

    return score


if __name__ == "__main__":
    _, cfg = parse_args()

    logger = create_logger(cfg)

    # print config
    logger.info(cfg.dump())
    # print device mode
    if torch.cuda.is_available():
        logger.info('GPU mode...')
    else:
        logger.info('CPU mode...')
    # training
    train(cfg, logger)


