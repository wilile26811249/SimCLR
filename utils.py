from os import path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn


class AverageMeter(object):
    def __init__(self,
        name: str,
        fmt: Optional[str] = ':f',
    ) -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,
        val: float,
        n: Optional[int] = 1
    ) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}:{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self,
        num_batches: int,
        meters: List[AverageMeter],
        prefix: Optional[str] = "",
        batch_info: Optional[str] = ""
    ) -> None:
        self.batch_fmster = self._get_batch_fmster(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.batch_info = batch_info

    def display(self, batch):
        self.info = [self.prefix + self.batch_info + self.batch_fmster.format(batch)]
        self.info += [str(meter) for meter in self.meters]
        print('\t'.join(self.info))

    def _get_batch_fmster(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk = (1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res