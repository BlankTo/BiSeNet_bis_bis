
import torch
from torch import nn
from torch.nn import functional as torch_functional
#import cv2
#import numpy as np
#import json

def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()

def get_one_hot(label, N, device):
    size = list(label.size())
    label = label.view(-1)
    #ones = torch.sparse.torch.eye(N).cuda()
    ones = torch.sparse.torch.eye(N)
    ones = ones.to(device)
    ones = ones.index_select(0, label.long())
    size.append(N)
    return ones.view(*size)

def get_boundary(gtmasks):

    laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32, device= gtmasks.device).reshape(1, 1, 3, 3).requires_grad_(False)
    # boundary_logits = boundary_logits.unsqueeze(1)
    boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1), laplacian_kernel, padding= 1)
    boundary_targets = boundary_targets.clamp(min= 0)
    boundary_targets[boundary_targets > 0.1] = 1
    boundary_targets[boundary_targets <= 0.1] = 0
    return boundary_targets


class DetailAggregateLoss(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__(*args, **kwargs)
        
        #self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        #self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]], dtype= torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))
        if device == 'cpu':
            self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.FloatTensor)
            self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]], dtype= torch.float32).reshape(1, 3, 1, 1).type(torch.FloatTensor))
        else:
            self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
            self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]], dtype= torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

        self.device = device

    def forward(self, boundary_logits, gtmasks):

        #print(boundary_logits.size())

        # boundary_logits = boundary_logits.unsqueeze(1)
        #boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        if self.device == 'cpu': boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, padding= 1)
        else: boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)

        boundary_targets = boundary_targets.clamp(min= 0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        #boundary_targets_x2 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        if self.device == 'cpu': boundary_targets_x2 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 2, padding= 1)
        else: boundary_targets_x2 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)

        boundary_targets_x2 = boundary_targets_x2.clamp(min= 0)
        
        #boundary_targets_x4 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        if self.device == 'cpu': boundary_targets_x4 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 4, padding= 1)
        else: boundary_targets_x4 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)

        boundary_targets_x4 = boundary_targets_x4.clamp(min= 0)

        #boundary_targets_x8 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        if self.device == 'cpu': boundary_targets_x8 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 8, padding= 1)
        else: boundary_targets_x8 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        
        boundary_targets_x8 = boundary_targets_x8.clamp(min= 0)
    
        boundary_targets_x8_up = torch_functional.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode= 'nearest')
        boundary_targets_x4_up = torch_functional.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode= 'nearest')
        boundary_targets_x2_up = torch_functional.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode= 'nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        #print(boundary_targets_x2_up.size())
        #print(boundary_targets_x4_up.size())
        #print(boundary_targets_x8_up.size())
        
        boundary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim= 1)
        
        #print((boundary_logits.size(), boundary_targets_pyramids.size()))

        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = torch_functional.conv2d(boundary_targets_pyramids, self.fuse_kernel)
        
        #print((boundary_logits.size(), boundary_targets_pyramid.size()))

        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0

        ##########################################################################
        ##########################################################################

        #this one is not from the original code, check in depth
        boundary_targets_pyramid = boundary_targets_pyramid.repeat(1, boundary_logits.shape[1], 1, 1)

        ##########################################################################
        ##########################################################################

        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = torch_functional.interpolate(boundary_logits, boundary_targets.shape[2:], mode= 'bilinear', align_corners= True)
        
        #print((boundary_logits.size(), boundary_targets_pyramid.size()))

        bce_loss = torch_functional.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boundary_targets_pyramid)
        return bce_loss,  dice_loss

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
                no_weight_decay_params += list(module.parameters())
        return no_weight_decay_params