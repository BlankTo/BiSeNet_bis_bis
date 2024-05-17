import torch
import logging

logger = logging.getLogger()

class Optimizer(object):
    def __init__(self,
                model,
                loss,
                lr0,
                momentum,
                weight_decay,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                ):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = 0
        weight_decay_params, no_weight_decay_params, lr_multiplier_weight_decay_params, lr_multiplier_no_weight_decay_params = model.get_params()
        loss_no_weight_decay_params = loss.get_params()

        # print(weight_decay_params)
        # print(no_weight_decay_params)
        # print(loss_no_weight_decay_params)
        # exit(0)

        param_list = [
                {'params': weight_decay_params},
                {'params': no_weight_decay_params, 'weight_decay': 0},
                {'params': lr_multiplier_weight_decay_params, 'lr_multiplier': True},
                {'params': lr_multiplier_no_weight_decay_params, 'weight_decay': 0, 'lr_multiplier': True},
                {'params': loss_no_weight_decay_params}]
                # {'params': loss_no_weight_decay_params, 'weight_decay': 0, 'lr': 0.000001}]

        self.optim = torch.optim.SGD(
                param_list,
                lr = lr0,
                momentum = momentum,
                weight_decay = weight_decay)
        self.warmup_factor = (self.lr0 / self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor**self.it)
        else:
            factor = (1 - (self.it - self.warmup_steps) / (self.max_iter - self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()
        for pg in self.optim.param_groups:
            if pg.get('lr_multiplier', False):
                pg['lr'] = self.lr * 10
            else:
                pg['lr'] = self.lr
        if self.optim.defaults.get('lr_multiplier', False):
            self.optim.defaults['lr'] = self.lr * 10
        else:
            self.optim.defaults['lr'] = self.lr
        self.it += 1
        self.optim.step()
        if self.it == self.warmup_steps+2:
            logger.info('==> warmup done, start to implement poly lr strategy')

    def zero_grad(self):
        self.optim.zero_grad()