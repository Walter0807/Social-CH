import numpy as np
import torch
import torch_dct as dct #https://github.com/zh217/torch-dct

import os

def disc_l2_loss(disc_value):
    
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(fake_disc_value, real_disc_value):
    kb = fake_disc_value.shape[0]
    ka = real_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la + lb


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dct_t(x):
    # do dct on the second last axis (t)
    x1 = torch.transpose(x, -1, -2)
    x1 = dct.dct(x1)
    x1 = torch.transpose(x1, -1, -2)
    return x1

def idct_t(x):
    # do idct on the second last axis (t)
    x1 = torch.transpose(x, -1, -2)
    x1 = dct.idct(x1)
    x1 = torch.transpose(x1, -1, -2)
    return x1

def dct_2d(x, scale=10):
    x1 = dct.dct(x) / scale
    return x1

def idct_2d(x, scale=10):
    x1 = dct.idct(x * scale) 
    return x1




def ensure_dir(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)