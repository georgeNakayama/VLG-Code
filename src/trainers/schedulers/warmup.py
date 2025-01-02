from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from typing import List, Optional
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class GradualWarmupScheduler(_LRScheduler):

    def __init__(self, optimizer: torch.optim.Optimizer, multiplier: float, total_epoch: int, after_scheduler: Optional[_LRScheduler]=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            return self.base_lrs
        return [base_lr * ( self.last_epoch + 1.)  / self.total_epoch for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class WarmupCosineAnnealingLR(GradualWarmupScheduler):
    def __init__(
        self, 
        optimizer: torch.optim.Optimizer, 
        epochs: int, 
        steps_per_epoch: int, 
        eta_min: float=0, 
        last_epoch: int=-1, 
        warmup_steps: int=100,
        multiplier: int=1
    ):
        after_scheduler = CosineAnnealingLR(optimizer, T_max=epochs*steps_per_epoch, eta_min=eta_min, last_epoch=last_epoch)
        super().__init__(optimizer, multiplier=multiplier, total_epoch=warmup_steps, after_scheduler=after_scheduler)
        
if __name__ == "__main__":
    lepoch = 200

    total_epoch = 5

    iters_per_epoch = int(145104 / (8 * 64)) 

    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.0002 / 8)
    consine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100 * iters_per_epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optim, multiplier=8, total_epoch=5 * iters_per_epoch, after_scheduler=consine_scheduler)

    a = []
    b = []

    for epoch in range(1, 100 * iters_per_epoch):
        scheduler.step(epoch)
        a.append(epoch)
        b.append(optim.param_groups[0]["lr"])
        # print(epoch, optim.param_groups[0]["lr"])
    
    plt.plot(a, b)
    plt.savefig('lr_sch_test1_iter.png')
    # plt.show()
