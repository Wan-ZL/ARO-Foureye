'''
Project     ：Drone-DRL-HT 
File        ：shared_classes.py
Author      ：Zelin Wan
Date        ：2/8/23
Description : 
'''

import torch


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class SharedSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        super(SharedSGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                                        weight_decay=weight_decay, nesterov=nesterov)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                if momentum != 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                # share in memory
                if momentum != 0:
                    state['momentum_buffer'].share_memory_()

class SharedReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False,
                 threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8):
        super(SharedReduceLROnPlateau, self).__init__(optimizer, mode=mode, factor=factor, patience=patience,
                                                       verbose=verbose, threshold=threshold,
                                                       threshold_mode=threshold_mode, cooldown=cooldown,
                                                       min_lr=min_lr, eps=eps)
        # State initialization
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['step'] = 0
                state['last_lr'] = group['lr']
                state['best'] = group['lr']
                state['num_bad_epochs'] = 0
                state['best_score'] = None

                # share in memory
                state['last_lr'].share_memory_()
                state['best'].share_memory_()
                state['num_bad_epochs'].share_memory_()
                state['best_score'].share_memory_()