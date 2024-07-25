# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer

from transformers.utils.versions import require_version
from einops import einsum

def attn_implementation(grad, exp_avg, exp_avg_sq, strategy, attn_impl, window = 0, step = 0, beta1 = 1.0, beta2 = 1.0):
    if attn_impl == "element":
        cat_exp_avg = torch.cat([each.unsqueeze(0) * beta1 for each in exp_avg] + [(1.0 - beta1) * grad.unsqueeze(0)], dim = 0)
        avg_attn_map = F.softmax((cat_exp_avg * grad.unsqueeze(0)), dim = 0)
        cat_exp_avg_sq = torch.cat([each.unsqueeze(0) * beta2 for each in exp_avg_sq] + [(1.0 - beta2) * (grad.unsqueeze(0)**2)], dim = 0)
        avg_sq_attn_map = F.softmax((cat_exp_avg_sq * (grad.unsqueeze(0)**2)), dim = 0)
        new_exp_avg = (avg_attn_map * cat_exp_avg).sum(0)
        new_exp_avg_sq = (avg_sq_attn_map * cat_exp_avg_sq).sum(0)
    elif attn_impl == "row":
        cat_exp_avg = torch.cat([each.unsqueeze(0) * beta1 for each in exp_avg] + [(1.0 - beta1) * grad.unsqueeze(0)], dim = 0)
        avg_attn_map = F.softmax(
            einsum(cat_exp_avg, grad, "t h d, h d -> h t")
            , dim = -1)
        cat_exp_avg_sq = torch.cat([each.unsqueeze(0) * beta2 for each in exp_avg_sq] + [(1.0 - beta2) * (grad.unsqueeze(0)**2)], dim = 0)
        avg_sq_attn_map = F.softmax(
            einsum(cat_exp_avg_sq,(grad**2), "t h d, h d -> h t")
            , dim = -1)
        new_exp_avg = einsum(avg_attn_map, cat_exp_avg, "h t, t h d -> h d")
        new_exp_avg_sq = einsum(avg_sq_attn_map, cat_exp_avg_sq, "h t, t h d -> h d")
    elif attn_impl == "matrix":
        shape = grad.shape
        cat_exp_avg = torch.cat([each.flatten().unsqueeze(0) * beta1 for each in exp_avg] + [(1.0 - beta1) * grad.flatten().unsqueeze(0)], dim = 0)
        avg_attn_map = F.softmax((grad.flatten().unsqueeze(0) @ cat_exp_avg.T), dim = 0)
        cat_exp_avg_sq = torch.cat([each.flatten().unsqueeze(0) * beta2 for each in exp_avg_sq] + [(1.0 - beta2) * (grad.flatten()**2).unsqueeze(0)], dim = 0)
        avg_sq_attn_map = F.softmax(((grad.flatten()**2).unsqueeze(0) @ cat_exp_avg_sq), dim = 0)
        new_exp_avg = torch.reshape((avg_attn_map @ cat_exp_avg).squeeze(), shape)
        new_exp_avg_sq = torch.reshape((avg_sq_attn_map @ cat_exp_avg_sq).squeeze(), shape)
    if strategy == "cascade":
        exp_avg[0].data = new_exp_avg.data
        exp_avg_sq[0].data = new_exp_avg_sq.data
        return exp_avg[0], exp_avg_sq[0]
    elif strategy == "window":
        idx = step%window
        exp_avg[idx].data = grad.data
        exp_avg_sq[idx].data = (grad**2).data
        return new_exp_avg, new_exp_avg_sq
    elif strategy == "cascade_window":
        idx = step%window
        exp_avg[idx].data = new_exp_avg.data
        exp_avg_sq[idx].data = new_exp_avg_sq.data
        return new_exp_avg, new_exp_avg_sq


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                
                #'strategy': ["cascade", "window", "cascade_window"]
                #'history': 2
                #'attn_implementation': ["element", "row", "column", "weight"]
                # State initialization
                if "exp_avg" not in state:
                    if "strategy" in group:
                        if group["strategy"] == "cascade":
                            state["exp_avg"] = [grad.clone()]
                            state["exp_avg_sq"] = [grad.clone()]
                        elif group["strategy"] == "window":
                            state["exp_avg"] = [torch.zeros_like(grad) for _ in range(group['history'])]
                            state["exp_avg_sq"] = [torch.zeros_like(grad) for _ in range(group['history'])]
                        elif group["strategy"] == "cascade_window":
                            state["exp_avg"] = [torch.zeros_like(grad) for _ in range(group['history'])]
                            state["exp_avg_sq"] = [torch.zeros_like(grad) for _ in range(group['history'])]
                    else:
                        state["exp_avg"] = [torch.zeros_like(grad)]
                        state["exp_avg_sq"] = [torch.zeros_like(grad)]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time

                if "strategy" not in group:
                    exp_avg = exp_avg[0]
                    exp_avg_sq = exp_avg_sq[0]
                    exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                else:
                    exp_avg, exp_avg_sq = attn_implementation(grad, 
                                                              exp_avg, 
                                                              exp_avg_sq, 
                                                              group["strategy"], 
                                                              group["attn_implementation"], 
                                                              group["history"], 
                                                              state["step"],
                                                              beta1,
                                                              beta2
                                                              )

                state["step"] += 1
                denom = exp_avg_sq.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        return loss