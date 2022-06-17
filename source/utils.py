# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-09-28
@File    : utils.py
"""

import os
import csv
import time
import math
import random
import logging
import numpy as np
from termcolor import colored

import torch
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, roc_auc_score


# -----------------------------------------------------------------------------
# Set seed for random, numpy, torch, cuda.
# -----------------------------------------------------------------------------
def seed_set(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Model resuming & checkpoint loading and saving.
# -----------------------------------------------------------------------------
def load_checkpoint(cfg, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {cfg.TRAIN.RESUME}....................")

    checkpoint = torch.load(cfg.TRAIN.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    best_epoch, best_auc = 0, 0.0
    if not cfg.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.defrost()
        cfg.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        cfg.freeze()
        logger.info(f"=> loaded successfully '{cfg.TRAIN.RESUME}' (epoch {checkpoint['epoch']})")
        if 'best_auc' in checkpoint:
            best_auc = checkpoint['best_auc']
        if 'best_epoch' in checkpoint:
            best_epoch = checkpoint['best_epoch']

    del checkpoint
    torch.cuda.empty_cache()
    return best_epoch, best_auc


def save_best_checkpoint(cfg, epoch, model, best_auc, best_epoch, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'best_auc': best_auc,
                  'best_epoch': best_epoch,
                  'epoch': epoch,
                  'config': cfg}

    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_path = os.path.join(ckpt_dir, f'best_ckpt.pth')
    torch.save(save_state, save_path)
    logger.info(f"best_ckpt saved !!!")


def load_best_result(cfg, model, logger):
    ckpt_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    best_ckpt_path = os.path.join(ckpt_dir, f'best_ckpt.pth')
    logger.info(f'Ckpt loading: {best_ckpt_path}')
    ckpt = torch.load(best_ckpt_path)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt['best_epoch']

    return model, best_epoch


# -----------------------------------------------------------------------------
# Log
# -----------------------------------------------------------------------------
def create_logger(cfg):
    # log name
    dataset_name = cfg.DATA.DATASET
    tag_name = cfg.TAG
    time_str = time.strftime("%Y-%m-%d")
    log_name = "{}_{}_{}.log".format(dataset_name, tag_name, time_str)

    # log dir
    log_dir = os.path.join(cfg.OUTPUT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = \
        colored('[%(asctime)s]', 'green') + \
        colored('(%(filename)s %(lineno)d): ', 'yellow') + \
        colored('%(levelname)-5s', 'magenta') + ' %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Data utils
# -----------------------------------------------------------------------------
def get_header(path):
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_task_names(path, use_compound_names=False):
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
def build_optimizer(cfg, model):
    params = model.parameters()

    opt_lower = cfg.TRAIN.OPTIMIZER.TYPE.lower()
    optimizer = None

    if opt_lower == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif opt_lower == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.TRAIN.OPTIMIZER.BASE_LR,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


# -----------------------------------------------------------------------------
# Lr_scheduler
# -----------------------------------------------------------------------------
def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.TRAIN.LR_SCHEDULER.FACTOR,
            patience=cfg.TRAIN.LR_SCHEDULER.PATIENCE,
            min_lr=cfg.TRAIN.LR_SCHEDULER.MIN_LR
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "noam":
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=[cfg.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS],
            total_epochs=[cfg.TRAIN.MAX_EPOCHS],
            steps_per_epoch=steps_per_epoch,
            init_lr=[cfg.TRAIN.LR_SCHEDULER.INIT_LR],
            max_lr=[cfg.TRAIN.LR_SCHEDULER.MAX_LR],
            final_lr=[cfg.TRAIN.LR_SCHEDULER.FINAL_LR]
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


class NoamLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch,
                 init_lr, max_lr, final_lr):

        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):

        return list(self.lr)

    def step(self, current_step=None):

        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


# -----------------------------------------------------------------------------
# Metric utils
# -----------------------------------------------------------------------------
def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets, preds):
    return mean_squared_error(targets, preds)


def get_metric_func(metric):

    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mean_absolute_error

    raise ValueError(f'Metric "{metric}" not supported.')

