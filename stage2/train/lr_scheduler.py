import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class ConstantScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr: float,
        verbose: bool = False
    ):
        self.lr = lr
        super(ConstantScheduler, self).__init__(optimizer=optimizer, verbose=verbose)


    def get_lr(self):
        return self.lr


    def step(self):
        for i in range(len(self.optimizer.param_groups)):
            self.print_lr(self.verbose, i, self.lr)
        self._step_count += 1



class CosineAnnealingAfterWarmUpScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        cycle_steps: int,
        max_lr: float,
        min_lr: float,
        damping_ratio: float,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.damping_ratio = damping_ratio
        super(CosineAnnealingAfterWarmUpScheduler, self).__init__(optimizer=optimizer, verbose=verbose)


    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) / self.warmup_steps * self._step_count
        else:
            if self.cycle_steps > self.warmup_steps:
                x = (self._step_count - self.warmup_steps) / (self.cycle_steps - self.warmup_steps) / 2 * math.pi
            else:
                x = (self._step_count - self.warmup_steps) / 2 * math.pi
            return self.min_lr + (self.max_lr - self.min_lr) * math.cos(x)


    def step(self):
        lr = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = lr
            self.print_lr(self.verbose, i, lr)
        self._step_count += 1



class CosineAnnealingAfterWarmUpAndHardRestartScheduler(CosineAnnealingAfterWarmUpScheduler):
    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return self.min_lr + (self.max_lr - self.min_lr) / self.warmup_steps * self._step_count
        elif (self._step_count - self.warmup_steps) % self.cycle_steps == 0:
            return self.max_lr * (1- self.damping_ratio * ((self._step_count - self.warmup_steps) // self.cycle_steps))
        else:
            x = ((self._step_count - self.warmup_steps) % self.cycle_steps) / (2 * self.cycle_steps) * math.pi
            return self.min_lr + (self.max_lr - self.min_lr) * math.cos(x)



# code from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annearing_with_warmup.py
class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < cycle_steps
        
        self.first_cycle_steps = cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]


    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
