import os
import sys

import torch
#from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_part_path = os.path.join(current_dir, 'dataset_part')
sys.path.append(dataset_part_path)
from dataset_part.cityscapes import CityScapes

model_part_path = os.path.join(current_dir, 'model_part')
sys.path.append(model_part_path)
from model_part.stages import BiSeNet

loss_part_path = os.path.join(current_dir, 'loss_part')
sys.path.append(loss_part_path)
from loss_part.loss import OhemCELoss
from loss_part.detail_loss import DetailAggregateLoss

optimizer_part_path = os.path.join(current_dir, 'optimizer_part')
sys.path.append(optimizer_part_path)
from optimizer_part.optimizer import Optimizer

path_to_res = ''


def train():

    mode = 'train'

    backbone_name = 'STDCNet813'
    data_path = '.\\dataset_part\\data'
    checkpoint_save_path = f'model_part\\checkpoints\\train_{backbone_name}\\result'

    pretrain_path = 'model_part\\checkpoints\\STDCNet813M_73.91.tar'
    #pretrain_path = ''

    if not os.path.exists(checkpoint_save_path): os.makedirs(checkpoint_save_path)

    # model params (?)

    n_classes = 19
    batch_size = 16
    use_boundary_16 = False
    use_boundary_8 = True
    use_boundary_4 = False
    use_boundary_2 = False

    use_conv_last = False
    checkpoint = None

    # optimizer params

    maxmIOU50 = 0.
    maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 60000
    save_iter_sep = 1000
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5

    cropsize = [1024, 512]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)    
    
    ds = CityScapes(data_path, cropsize= cropsize, mode= mode, randomscale= randomscale)
    #sampler = RandomSampler(ds)
    sampler = SequentialSampler(ds)
    dl = DataLoader(ds, batch_size= batch_size, shuffle= False, sampler= sampler, pin_memory= False, drop_last= True)

    ds_val = CityScapes(data_path, mode= 'val', randomscale= randomscale)
    #sampler_val = RandomSampler(ds_val)
    sampler_val = SequentialSampler(ds_val)
    dl_val = DataLoader(ds_val, batch_size= 2, shuffle= False, sampler= sampler_val, drop_last= False)

    ## model

    label_to_ignore = 255

    net = BiSeNet(
        backbone_name= backbone_name,
        n_classes= n_classes,
        pretrain_model= pretrain_path,
        use_boundary_2= use_boundary_2,
        use_boundary_4= use_boundary_4,
        use_boundary_8= use_boundary_8,
        use_boundary_16= use_boundary_16,
        use_conv_last= use_conv_last
        )
    
    if checkpoint is not None:
        net.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    #net.cuda()
    net.train() # set the net to train mode

    #net = nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank, ], output_device = args.local_rank, find_unused_parameters=True)

    score_thres = 0.7
    n_min = (batch_size * cropsize[0] * cropsize[1]) // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, label_to_ignore= label_to_ignore)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, label_to_ignore= label_to_ignore)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, label_to_ignore= label_to_ignore)
    boundary_loss_func = DetailAggregateLoss()

    ## optimizer

    optimizer = Optimizer(
            #model = net.module,
            model = net,
            loss = boundary_loss_func,
            lr0 = lr_start,
            momentum = momentum,
            weight_decay = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    
    ## train loop (also evaluation.py)

    #TODO

    ## save model

    #TODO

train()