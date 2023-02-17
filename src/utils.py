import pickle
import numpy as np
import torch
import torch.nn as nn
import random

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.history = []
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    
    # def append_loss_value(self, val):
    #     self.history.append(val)

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print('Loaded data from ', path)
    return data

def save_pkl(file, path):
    with open(path, 'wb+') as f:
        pickle.dump(file, f)
    print('Saved to ', path)

def get_parameters_number(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return print(f'Model has {params} params')

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def l2_normalize(X):
    """
    L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def make_train_test_dev_dfs(dataf):
    train  = dataf[dataf['eval_status'] == 'train']
    test  = dataf[dataf['eval_status'] == 'test']
    dev  = dataf[dataf['eval_status'] == 'dev']

    return train, test, dev

def get_dt_string():
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d_%m_%Y_%Hh_%Mm")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_model(config, model_class):
    model = model_class()
    model.load_state_dict(torch.load(config["model_path"], map_location=config["device"]))
    model.eval()
    return model