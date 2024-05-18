import os
import sys
import time

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

evaluation_part_path = os.path.join(current_dir, 'evaluation_part')
sys.path.append(evaluation_part_path)
from evaluation_part.evaluation import MscEvalV0

def train():

    mode = 'train'

    backbone_name = 'STDCNet813'
    data_path = os.path.join('dataset_part') #'.\\dataset_part\\data'
    checkpoint_save_path = os.path.join('model_part', 'checkpoints', f'train_{backbone_name}', 'result') #f'model_part\\checkpoints\\train_{backbone_name}\\result'

    pretrain_path = os.path.join('model_part', 'checkpoints', 'STDCNet813M_73.91.tar') #'model_part\\checkpoints\\STDCNet813M_73.91.tar'
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
    
    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    #starting_time = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for i in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0] == batch_size: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        #im = im.cuda()
        #lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optimizer.zero_grad()


        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(im)
        
        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
            print(len(net(im)))
            out, out16, out32, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
            out, out16, out32 = net(im)

        lossp = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss3 = criteria_32(out32, lb)
        
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.
        
        
        if use_boundary_2: 
            # if dist.get_rank()==0:
            #     print('use_boundary_2')
            boundery_bce_loss2,  boundery_dice_loss2 = boundary_loss_func(detail2, lb)
            boundery_bce_loss += boundery_bce_loss2
            boundery_dice_loss += boundery_dice_loss2
        
        if use_boundary_4:
            # if dist.get_rank()==0:
            #     print('use_boundary_4')
            boundery_bce_loss4,  boundery_dice_loss4 = boundary_loss_func(detail4, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4

        if use_boundary_8:
            # if dist.get_rank()==0:
            #     print('use_boundary_8')
            boundery_bce_loss8,  boundery_dice_loss8 = boundary_loss_func(detail8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8

        loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss
        
        loss.backward()
        optimizer.step()

        loss_avg.append(loss.item())

        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())

        ## print training log message
        if (i + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optimizer.lr
            ed = time.time()
            #t_intv, glob_t_intv = ed - starting_time, ed - glob_st
            #eta = int((max_iter - it) * (glob_t_intv / it))
            #eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                f'it: {i + 1}/{max_iter}',
                f'lr: {lr:4f}',
                f'loss: {loss_avg:.4f}',
                f'boundery_bce_loss: {loss_boundery_bce_avg:.4f}',
                f'boundery_dice_loss: {loss_boundery_dice_avg:.4f}',
                #f'eta: {eta}',
            ])
            
            print(msg)
            #logger.info(msg)
            
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed
            # print(boundary_loss_func.get_params())

        if ((i + 1) % save_iter_sep) == 0:# and i != 0:
            
            ## model
            #logger.info('evaluating the model ...')
            #logger.info('setup and restore model')

            print('evaluating the model ...')
            print('setup and restore model')
            
            net.eval() # set the net in evaluation mode

            ## evaluator

            #logger.info('compute the mIOU')
            print('compute the mIOU')

            with torch.no_grad():
                single_scale1 = MscEvalV0()
                mIOU50 = single_scale1(net, dl_val, n_classes)

                single_scale2= MscEvalV0(scale=0.75)
                mIOU75 = single_scale2(net, dl_val, n_classes)


            save_pth = os.path.join(checkpoint_save_path, f'model_iter{(i + 1)}_mIOU50_{str(round(mIOU50,4))}_mIOU75_{str(round(mIOU75,4))}.pth')
            
            #state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            state = net.state_dict()

            #if dist.get_rank()==0: torch.save(state, save_pth)
            torch.save(state, save_pth)

            #logger.info('training iteration {}, model saved to: {}'.format(i + 1, save_pth))
            print(f'training iteration {(i + 1)}, model saved to: {save_pth}')

            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = os.path.join(checkpoint_save_path, 'model_maxmIOU50.pth'.format(i + 1))
                #state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                state = net.state_dict()
                #if dist.get_rank()==0: torch.save(state, save_pth)
                torch.save(state, save_pth)
                    
                #logger.info('max mIOU model saved to: {}'.format(save_pth))
                print(f'max mIOU model saved to: {save_pth}')
            
            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
                save_pth = os.path.join(checkpoint_save_path, 'model_maxmIOU75.pth'.format(i + 1))
                #state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                state = net.state_dict()
                #if dist.get_rank()==0: torch.save(state, save_pth)
                torch.save(state, save_pth)
                #logger.info('max mIOU model saved to: {}'.format(save_pth))
                print(f'max mIOU model saved to: {save_pth}')
            
            #logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))
            #logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))
            print(f'mIOU50 is: {mIOU50}, mIOU75 is: {mIOU75}')
            print(f'maxmIOU50 is: {maxmIOU50}, maxmIOU75 is: {maxmIOU75}.')

            net.train()

    ## save model

    save_pth = os.path.join(checkpoint_save_path, 'model_final.pth')
    #net.cpu()
    #state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    state = net.state_dict()
    #if dist.get_rank()==0: torch.save(state, save_pth)
    torch.save(state, save_pth)
    #logger.info('training done, model saved to: {}'.format(save_pth))
    print('training done, model saved to: {save_pth}')
    print('epoch: ', epoch)

train()