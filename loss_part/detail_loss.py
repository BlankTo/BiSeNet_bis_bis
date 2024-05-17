
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

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    #ones = torch.sparse.torch.eye(N).cuda()
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label.long())   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
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
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__(*args, **kwargs)
        
        #self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)
        self.laplacian_kernel = torch.tensor([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype= torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.FloatTensor)
        
        #self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]], dtype= torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]], dtype= torch.float32).reshape(1, 3, 1, 1).type(torch.FloatTensor))

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        #boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, padding= 1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        #boundary_targets_x2 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 2, padding= 1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        #boundary_targets_x4 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 4, padding= 1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        #boundary_targets_x8 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = torch_functional.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride= 8, padding= 1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = torch_functional.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode= 'nearest')
        boundary_targets_x4_up = torch_functional.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode= 'nearest')
        boundary_targets_x2_up = torch_functional.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode= 'nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim= 1)
        
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = torch_functional.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = torch_functional.interpolate(boundary_logits, boundary_targets.shape[2:], mode= 'bilinear', align_corners= True)
        
        bce_loss = torch_functional.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss,  dice_loss

    def get_params(self):
        weight_decay_params, no_weight_decay_params = [], []
        for name, module in self.named_modules():
                no_weight_decay_params += list(module.parameters())
        return no_weight_decay_params