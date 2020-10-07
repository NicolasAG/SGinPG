"""Optimizer utilities."""

import math
import numpy as np
import torch
from torch.optim import Optimizer


class Adam(Optimizer):
    """Adam optimizer.

    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.
    This fixes the way weight decay interacts with Adam:
    https://openreview.net/pdf?id=rk6qdGgCZ
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    ):
        """Init params."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0  # torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def __setstate__(self, state):
        """Set optimizer state."""
        super().__setstate__(state)

    def step(self, closure=None):
        """Step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']  # .item()
                bias_correction2 = 1 - beta2 ** state['step']  # .item()
                step_size = group['lr'] * math.sqrt(bias_correction2) \
                    / bias_correction1

                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class AdamCosineWithWarmup(Adam):
    """Adam with fixed weight decay and cosine schedule.

    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(
        self, params, lr=2.5e-4, betas=(0.9, 0.999), eps=1e-6,
        weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
        min_lr=1e-9, init_period=1000000, period_mult=1, lr_shrink=0.75
    ):
        """Init params."""
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, apply cosine scheduler
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, n_current_steps):
        """Learning rate adaptive logic."""
        self.n_current_steps = n_current_steps
        if n_current_steps < self.warmup_updates:
            self.cur_lr = self.warmup_init_lr + n_current_steps * self.lr_step
            return self.cur_lr
        else:
            t = n_current_steps - self.warmup_updates
            if self.period_mult == 1:
                pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - (self.period * pid)
            else:
                pid = math.floor(math.log(
                    1 - t / self.period * (1 - self.period_mult),
                    self.period_mult
                ))
                t_i = self.period * (self.period_mult ** pid)
                t_curr = t - (1 - self.period_mult ** pid) \
                    / (1 - self.period_mult) * self.period
            lr_shrink = self.lr_shrink ** pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            self.cur_lr = min_lr + 0.5 * (max_lr - min_lr) * \
                (1 + math.cos(math.pi * t_curr / t_i))
            return self.cur_lr

    def step(self, closure=None):
        """Step."""
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(
                param_group['num_updates']
            )
