import numpy as np
import time
import sys
import itertools 
import torch
import torch.utils.data
from torch.autograd import Variable
from torch.nn import functional as F
import random
from torch.distributions import Beta
import math
from PIL import Image
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


# **********    

class Logger(object):

    def __init__(self, mode, length, calculate_mean=False):
        self.mode = mode
        self.length = length
        self.calculate_mean = calculate_mean
        if self.calculate_mean:
            self.fn = lambda x, i: x / (i + 1)
        else:
            self.fn = lambda x, i: x
        self.fn_no_mean = lambda x, i: x

    def __call__(self, loss, cls_loss, metrics, i):
        track_str = '\r{} | {:5d}/{:<5d}| '.format(self.mode, i + 1, self.length)
        loss_str = 'loss: {:5.4f} | '.format(self.fn(loss, i))
        cls_loss = 'cls_loss: {:5.4f} | '.format(self.fn_no_mean(cls_loss, i))
        metric_str = ' | '.join('{}: {:9.4f}'.format(k, self.fn(v, i)) for k, v in metrics.items())
        print(track_str + loss_str + cls_loss + metric_str + ' ', end='')
        if i + 1 == self.length:
            print('')

# **********

class BatchTimer(object):
    
    """Batch timing class.
    Use this class for tracking training and testing time/rate per batch or per sample.
    
    Keyword Arguments:
        rate {bool} -- Whether to report a rate (batches or samples per second) or a time (seconds
            per batch or sample). (default: {True})
        per_sample {bool} -- Whether to report times or rates per sample or per batch.
            (default: {True})
    """

    def __init__(self, rate=True, per_sample=True):
        self.start = time.time()
        self.end = None
        self.rate = rate
        self.per_sample = per_sample

    def __call__(self, y_pred, y):
        self.end = time.time()
        elapsed = self.end - self.start
        self.start = self.end
        self.end = None

        if self.per_sample:
            elapsed /= len(y_pred)
        if self.rate:
            elapsed = 1 / elapsed

        return torch.tensor(elapsed)

# **********

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()

# **********

def run_train(feature_extractor, generator, feat_fc, hash_fc, data_loader,
                epoch = 1, net_params = None, loss_fn = None, optimizer = None,
                scheduler = None, batch_metrics = {'time': BatchTimer()},
                show_running = True, device = 'cuda:0', writer = None):
    
    mode = 'Train'
    iter_max = len(data_loader)
    logger = Logger(mode, length = iter_max, calculate_mean = show_running)
    
    loss = 0
    metrics = {}
    
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        embeddings = feature_extractor(inputs)
        embeddings_gen = generator(embeddings, labels, training=True)

        pred = feat_fc(embeddings, labels)
        pred_gen = hash_fc(embeddings_gen, labels)

        loss_ce = loss_fn['loss_ce'](pred, labels) + loss_fn['loss_ce'](pred_gen, labels)

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(pred, labels).detach().cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]
            
        if writer is not None:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_ce.detach().cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_ce.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, loss_ce, metrics, batch_idx)
        else:
            logger(loss_batch, metrics_batch, batch_idx)
    
    # *** ***

    if scheduler is not None:
        scheduler.step()

    loss = loss / (batch_idx + 1)
    metrics = {k: v / (batch_idx + 1) for k, v in metrics.items()}
    
    return metrics, loss
