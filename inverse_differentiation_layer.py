#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 13.02.24
from typing import Any, Callable

import torch
from torch.autograd import Function
from torch.autograd.functional import jacobian as autograd_jacobian


class ImplicitInverseLayer(Function):
    """
    Implements an

    The gradients for the backward pass are computed using the inverse function theorem. The theorem states that the
    jacobian of an inverse function can be obtained by:

    J(g)(x) = [J(f(y))]^-1, where g is the right inverse of the forward function f, s.t. f(g(x)) = x.

    :reference: Krantz and Parks, The Implicit Function Theorem, 2002
    :reference: https://mathweb.ucsd.edu/~nwallach/inverse%5B1%5D.pdf
    :reference: https://pytorch.org/docs/stable/notes/extending.html
    :reference: https://github.com/jcjohnson/pytorch-examples#pytorch-defining-new-autograd-functions
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, inverse_func: Callable, forward_jacobian: torch.Tensor) -> Any:
        """
        The forward pass of a function "inverse_func" which is the inverse to another, known, forward function.

        :param ctx: The context to save tensors for the backward pass.
        :param x: The input to the module.
        :param inverse_func: The forward pass of this function. It needs to be an inverse function that does not allow
            for an easy computation of gradients with backprop (or else this wrapper would be useless).
        :param forward_jacobian: The Jacobian of the forward function to inverse_func. Used for an implicit
            differentiation in the backwards pass.
        """
        ctx.save_for_backward(x, forward_jacobian)
        return inverse_func(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        """
        The backward pass of the module.

        Returns the gradients of the input with respect to the loss.
        Also returns None, None, because we don't (or cannot) compute gradients for the inverse function or the
            forward jacobian.
        """
        x, jacobian = ctx.saved_tensors
        j_inv = torch.inverse(jacobian)
        grad_input = torch.einsum('bj,bij->bj', grad_output, j_inv)
        return grad_input, None, None


def batch_jacobian(func: Callable, inputs: torch.Tensor, reduce_with_sum: bool = True, **kwargs) -> torch.Tensor:
    """
    Computes the Jacobian of func w.r.t. inputs, assuming that the first dimension is a batch dimension.

    Explanation: torch.autograd is not aware of a thing like "batch" size and a naive use of the
        autograd.functional.jacobian function would result in an overhead by computing the gradient of each
        samples outbut w.r.t to each samples input, which is not what we want. This function computes the
        Jacobian indidivually for each sample in the batch and stacks them together.

    This could be done with a for loop, but we apply a little trick to make it more efficient: We sum the inputs
        along the batch dimension and then compute the Jacobian of the sum. As long as the inputs are independent,
        this is equivalent to the Jacobian of the individual inputs.


    :source: https://discuss.pytorch.org/t/jacobian-functional-api-batch-respecting-jacobian/84571/5

    :param func: The function to compute the Jacobian for.
    :param inputs: The inputs to the function. The first dimension must be a batch dimension.
    :param reduce_with_sum: Whether to reduce the Jacobian by summing over the batch dimension. If False, the
        full Jacobian will be computed and the diagonal will be returned, which is less efficient.
    :param kwargs: Anything that should be passed to torch.autograd.functional.jacobian.

    :return: The Jacobian of func w.r.t. inputs.
    """
    if reduce_with_sum:
        f_sum = lambda x: torch.sum(func(x), axis=0)
        jac = autograd_jacobian(f_sum, inputs, **kwargs)
        batch_dim = len(jac.shape) - len(inputs.shape)
        return jac.permute(batch_dim, *range(batch_dim), *range(batch_dim + 1, len(jac.shape)))

    jac_full = autograd_jacobian(func, inputs, **kwargs)
    jac_diag = jac_full.diagonal(dim1=0, dim2=-len(inputs.shape))  # extract the diagonal only
    return jac_diag.permute(-1, *range(len(jac_diag.shape) - 1))  # move the batch dimension to the front


def implicit_inverse_layer(x: torch.Tensor, inverse_func: Callable, forward_jacobian: torch.Tensor) -> torch.Tensor:
    """
    A wrapper around the ImplicitInverseLayer that allows for a more convenient usage.

    :param x: The input to the module.
    :param inverse_func: The forward pass of this function. It needs to be an inverse function that does not allow
        for an easy computation of gradients with backprop (or else this wrapper would be useless).
    :param forward_jacobian: The Jacobian of the forward function to inverse_func. Used for an implicit
        differentiation in the backwards pass.
    """
    return ImplicitInverseLayer.apply(x, inverse_func, forward_jacobian)
