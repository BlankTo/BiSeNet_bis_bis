import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from loss_utils import enet_weighing
import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, label_to_ignore= 255, device= 'cpu', *args, **kwargs):
        super(OhemCELoss, self).__init__(*args, **kwargs)
        #self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.thresh = -torch.log(torch.tensor(thresh, dtype= torch.float))
        self.thresh = self.thresh.to(device)
        self.n_min = n_min
        self.label_to_ignore = label_to_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index= label_to_ignore, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class WeightedOhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, num_classes, label_to_ignore= 255, device= 'cpu', *args, **kwargs):
        super(WeightedOhemCELoss, self).__init__(*args, **kwargs)
        #self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.thresh = -torch.log(torch.tensor(thresh, dtype= torch.float))
        self.thresh = self.thresh.to(device)
        self.n_min = n_min
        self.label_to_ignore = label_to_ignore
        self.num_classes = num_classes
        # self.criteria = nn.CrossEntropyLoss(ignore_index=label_to_ignore, reduction='none')
        self.device = device

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        #criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(), ignore_index=self.label_to_ignore, reduction='none')
        criteria = nn.CrossEntropyLoss(weight= enet_weighing(labels, self.num_classes), ignore_index= self.label_to_ignore, reduction= 'none')
        criteria = criteria.to(self.device)
        loss = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, label_to_ignore= 255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__(*args, **kwargs)
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index= label_to_ignore)

    def forward(self, logits, labels):
        scores = torch_functional.softmax(logits, dim= 1)
        factor = torch.pow((1. - scores), self.gamma)
        log_score = torch_functional.log_softmax(logits, dim= 1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss