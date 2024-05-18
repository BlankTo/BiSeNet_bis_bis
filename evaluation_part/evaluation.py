#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import math
import torch
import numpy as np
from tqdm import tqdm
#import torch.nn as nn

from torch.utils.data import DataLoader
import torch.nn.functional as torch_functional

from dataset_part.cityscapes import CityScapes
from model_part.stages import BiSeNet



class MscEvalV0(object):

    def __init__(self, scale= 0.5, label_to_ignore= 255):
        self.label_to_ignore = label_to_ignore
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        #hist = torch.zeros(n_classes, n_classes).cuda().detach()
        hist = torch.zeros(n_classes, n_classes)
        #if dist.is_initialized() and dist.get_rank() != 0:
        #    diter = enumerate(dl)
        #else:
        #    diter = enumerate(tqdm(dl))
        diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:

            #N, _, Height, Width = label.shape

            #label = label.squeeze(1).cuda()
            label = label.squeeze(1)
            size = label.size()[-2:]

            #imgs = imgs.cuda()
            imgs = imgs

            N, C, Height, Width = imgs.size()
            new_height_width = [int(Height * self.scale), int(Width * self.scale)]

            imgs = torch_functional.interpolate(imgs, new_height_width, mode= 'bilinear', align_corners= True)

            logits = net(imgs)[0]
  
            logits = torch_functional.interpolate(logits, size= size, mode= 'bilinear', align_corners= True)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            keep = label != self.label_to_ignore
            hist += torch.bincount((label[keep] * n_classes + preds[keep]), minlength= n_classes ** 2).view(n_classes, n_classes).float()
        #if dist.is_initialized():
        #    dist.all_reduce(hist, dist.ReduceOp.SUM)
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        miou = ious.mean()
        return miou.item()
# '.\\model_part\\checkpoints' # '.\\dataset_part\\data'
def evaluatev0(checkpoint_save_path= os.path.join('model_part', 'checkpoints'), data_path= os.path.join('dataset_part', 'data'), backbone_name= 'No model selected', scale= 0.75, use_boundary_2= False, use_boundary_4= False, use_boundary_8= False, use_boundary_16= False, use_conv_last= False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)

    ## dataset
    batch_size = 5
    dsval = CityScapes(data_path, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batch_size,
                    shuffle = False,
                    drop_last = False)

    n_classes = 19
    print("backbone:", backbone_name)
    net = BiSeNet(
        backbone= backbone_name,
        n_classes= n_classes,
        use_boundary_2= use_boundary_2,
        use_boundary_4= use_boundary_4,
        use_boundary_8= use_boundary_8,
        use_boundary_16= use_boundary_16,
        use_conv_last= use_conv_last,
        )
    
    net.load_state_dict(torch.load(checkpoint_save_path))
    #net.cuda()
    net.eval()
    

    with torch.no_grad():
        single_scale = MscEvalV0(scale= scale)
        mIOU = single_scale(net, dl, 19)
    #logger = logging.getLogger()
    #logger.info('mIOU is: %s\n', mIOU)
    print('mIOU is: %s\n', mIOU)

class MscEval(object):
    def __init__(self,
            model,
            dataloader,
            scales= [0.5, 0.75, 1, 1.25, 1.5, 1.75],
            n_classes= 19,
            label_to_ignore= 255,
            cropsize= 1024,
            flip= True,
            ):
        self.scales = scales
        self.n_classes = n_classes
        self.label_to_ignore = label_to_ignore
        self.flip = flip
        self.cropsize = cropsize
        ## dataloader
        self.dl = dataloader
        self.net = model


    def pad_tensor(self, inten, size):
        N, C, Height, Width = inten.size()
        #outten = torch.zeros(N, C, size[0], size[1]).cuda()
        outten = torch.zeros(N, C, size[0], size[1])
        outten.requires_grad = False
        margin_h, margin_w = (size[0] - Height), (size[1] - Width)
        hst, hed = (margin_h // 2), (margin_h // 2 + Height)
        wst, wed = (margin_w // 2), (margin_w // 2 + Width)
        outten[:, :, hst:hed, wst:wed] = inten
        return outten, [hst, hed, wst, wed]


    def eval_chip(self, crop):
        with torch.no_grad():
            out = self.net(crop)[0]
            prob = torch_functional.softmax(out, 1)
            if self.flip:
                crop = torch.flip(crop, dims= (3,))
                out = self.net(crop)[0]
                out = torch.flip(out, dims= (3,))
                prob += torch_functional.softmax(out, 1)
            prob = torch.exp(prob)
        return prob


    def crop_eval(self, im):
        cropsize = self.cropsize
        stride_rate = 5/6.
        N, C, Height, Width = im.size()
        long_size, short_size = (Height, Width) if Height > Width else (Width, Height)
        if long_size < cropsize:
            im, indices = self.pad_tensor(im, (cropsize, cropsize))
            prob = self.eval_chip(im)
            prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        else:
            stride = math.ceil(cropsize * stride_rate)
            if short_size < cropsize:
                if Height < Width:
                    im, indices = self.pad_tensor(im, (cropsize, Width))
                else:
                    im, indices = self.pad_tensor(im, (Height, cropsize))
            N, C, Height, Width = im.size()
            n_x = math.ceil((Width - cropsize) / stride) + 1
            n_y = math.ceil((Height - cropsize) / stride) + 1
            #prob = torch.zeros(N, self.n_classes, Height, Width).cuda()
            prob = torch.zeros(N, self.n_classes, Height, Width)
            prob.requires_grad = False
            for iy in range(n_y):
                for ix in range(n_x):
                    hed, wed = min(Height, stride * iy + cropsize), min(Width, stride * ix + cropsize)
                    hst, wst = hed - cropsize, wed - cropsize
                    chip = im[:, :, hst:hed, wst:wed]
                    prob_chip = self.eval_chip(chip)
                    prob[:, :, hst:hed, wst:wed] += prob_chip
            if short_size < cropsize:
                prob = prob[:, :, indices[0]:indices[1], indices[2]:indices[3]]
        return prob


    def scale_crop_eval(self, im, scale):
        N, C, Height, Width = im.size()
        new_hw = [int(Height * scale), int(Width * scale)]
        im = torch_functional.interpolate(im, new_hw, mode= 'bilinear', align_corners= True)
        prob = self.crop_eval(im)
        prob = torch_functional.interpolate(prob, (Height, Width), mode= 'bilinear', align_corners= True)
        return prob


    def compute_hist(self, pred, lb):
        n_classes = self.n_classes
        ignore_idx = self.label_to_ignore
        keep = np.logical_not(lb == ignore_idx)
        merge = pred[keep] * n_classes + lb[keep]
        hist = np.bincount(merge, minlength= n_classes**2)
        hist = hist.reshape((n_classes, n_classes))
        return hist


    def evaluate(self):
        ## evaluate
        n_classes = self.n_classes
        hist = np.zeros((n_classes, n_classes), dtype= np.float32)
        dloader = tqdm(self.dl)
        #if dist.is_initialized() and not dist.get_rank()==0:
        #    dloader = self.dl
        for i, (imgs, label) in enumerate(dloader):
            N, _, Height, Width = label.shape
            probs = torch.zeros((N, self.n_classes, Height, Width))
            probs.requires_grad = False
            #imgs = imgs.cuda()
            imgs = imgs
            for sc in self.scales:
                # prob = self.scale_crop_eval(imgs, sc)
                prob = self.eval_chip(imgs)
                #probs += prob.detach().cpu()
                probs += prob.detach()
            probs = probs.data.numpy()
            preds = np.argmax(probs, axis= 1)

            hist_once = self.compute_hist(preds, label.data.numpy().squeeze(1))
            hist = hist + hist_once
        IOUs = np.diag(hist) / (np.sum(hist, axis= 0) + np.sum(hist, axis= 1) - np.diag(hist))
        mIOU = np.mean(IOUs)
        return mIOU

# '.\\model_part\\checkpoints' # '.\\dataset_part\\data'
def evaluate(checkpoint_save_path= os.path.join('model_part', 'checkpoints'), data_path= os.path.join('dataset_part', 'data')):

    ## model
    print('\n')
    print('===='*20)
    print('evaluating the model ...\n')
    print('setup and restore model')
    n_classes = 19
    net = BiSeNet(n_classes= n_classes)

    net.load_state_dict(torch.load(checkpoint_save_path))
    #net.cuda()
    net.eval()

    ## dataset
    batchsize = 5
    dsval = CityScapes(data_path, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    drop_last = False)

    ## evaluator
    print('compute the mIOU')
    evaluator = MscEval(net, dl, scales= [1], flip= False)

    ## eval
    mIOU = evaluator.evaluate()
    print('mIOU is: {:.6f}'.format(mIOU))