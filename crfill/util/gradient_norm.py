from __future__ import annotations
import math
import torch


def weight_norm(
    parameters: Union[Iterable[torch.Tensor], torch.Tensor],
    norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Compute global norm of an iterable of parameters.
    The norm is computed over all tensors together, as if they were
    concatenated into a single vector. This code is based on
    torch.nn.utils.clip_grad_norm_().
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0.0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def gradient_norm(
    parameters: Union[Iterable[torch.Tensor], torch.Tensor],
    norm_type: Optional[Union[float, int]] = 2,
) -> float:
    """Compute global norm of an iterable of parameters.
    The norm is computed over all tensors together, as if they were
    concatenated into a single vector. This code is based on
    torch.nn.utils.clip_grad_norm_().
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int, optional): type of the used p-norm. Can be
            ``'inf'`` for infinity norm.
    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    if norm_type == math.inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        parameters = [
            p for p in parameters if p.grad is not None and p.requires_grad
        ]  # gradients can be None
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    return total_norm
