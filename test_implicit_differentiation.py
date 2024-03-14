#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 14.03.24
import unittest

import torch

from inverse_differentiation_layer import batch_jacobian, implicit_inverse_layer


class ImplicitDifferentiation(unittest.TestCase):
    """
    Tests implicit differentiation utilities.

    The computations of gradients, depending on the exact input and function, might be affected by inaccuracies due to
    the finite precision of floating point numbers. Therefore, the tolerances for the tests are comparably large.
    Still, these tests would never pass even with these tolerances if the functions were not implemented correctly.
    """

    atol = .01
    rtol = .01

    @staticmethod
    def loss(x: torch.Tensor):
        """Simple loss function for testing."""
        return torch.sum(x)

    def test_batch_jacobian(self, b=64, max_dim=10):
        funcs = (torch.cos, lambda x: torch.sum(x, dim=-1))

        for dim in range(1, max_dim + 1):
            x = torch.rand((b, dim), requires_grad=True)
            for func in funcs:
                jac = batch_jacobian(func, x)
                jac_slow = batch_jacobian(func, x, reduce_with_sum=False)
                self.assertTrue(torch.allclose(jac, jac_slow, atol=self.atol, rtol=self.rtol))

    def test_inverse_layer(self, b=64):
        forward = {
            'sin': torch.sin,
            'cos': torch.cos,
            'x^2': lambda x: x ** 2,
            'two_angles': lambda x: torch.stack([torch.cos(x[:, 1]), torch.sin(x[:, 0])], dim=-1)
        }
        inverse = {
            'sin': torch.asin,
            'cos': torch.acos,
            'x^2': lambda x: torch.sqrt(x),
            'two_angles': lambda x: torch.stack([torch.asin(x[:, 1]), torch.acos(x[:, 0])], dim=-1),
        }
        num_args = {
            'sin': 1,
            'cos': 1,
            'x^2': 1,
            'two_angles': 2,
        }

        for name, inv in inverse.items():
            # Start with the most natural use case
            x = torch.rand((b, num_args[name]), requires_grad=True)
            y_with_grad = implicit_inverse_layer(x, inv, forward_func=forward[name])
            gradients_implicit = torch.autograd.grad(self.loss(y_with_grad), x)[0]

            # Ground truth gradients, computed with autograd (which is not possible in a real world application)
            y_autograd = inv(x)
            gradients_gt = torch.autograd.grad(self.loss(y_autograd), x)[0]

            self.assertTrue(torch.allclose(gradients_gt, gradients_implicit, atol=self.atol, rtol=self.rtol))

        for name, fw in forward.items():
            # Make sure it works the other way around as well
            x = torch.rand((b, num_args[name]), requires_grad=True)
            y_with_grad = implicit_inverse_layer(x, fw, forward_func=inverse[name])
            gradients_implicit = torch.autograd.grad(self.loss(y_with_grad), x)[0]

            # Ground truth gradients, computed with autograd (which is not possible in a real world application)
            y_autograd = fw(x)
            gradients_gt = torch.autograd.grad(self.loss(y_autograd), x)[0]

            self.assertTrue(torch.allclose(gradients_gt, gradients_implicit, atol=self.atol, rtol=self.rtol))

        for name, inv in inverse.items():
            # Test the use case where a Jacobian is provided externally
            x = torch.rand((b, num_args[name]), requires_grad=True)

            with torch.no_grad():
                y = inv(x)

            J = batch_jacobian(forward[name], y)
            y_with_grad = implicit_inverse_layer(x, inv, forward_jacobian=J)
            gradients_implicit = torch.autograd.grad(self.loss(y_with_grad), x)[0]

            # Ground truth gradients, computed with autograd (which is not possible in a real world application)
            y_autograd = inv(x)
            gradients_gt = torch.autograd.grad(self.loss(y_autograd), x)[0]

            self.assertTrue(torch.allclose(gradients_gt, gradients_implicit, atol=self.atol, rtol=self.rtol))

        for name, fw in forward.items():
            # Test that the functions are actually inverses of each other
            x = torch.rand((b, num_args[name]), requires_grad=True)
            y_forward = fw(x)
            forward_jacobian = batch_jacobian(fw, x)

            inv = inverse[name]
            y_inv = inv(y_forward)
            self.assertTrue(torch.allclose(y_inv, x, atol=self.atol, rtol=self.rtol))

            # Ground truth gradients, computed with autograd (which is not possible in a real world application)
            gradients_gt = torch.autograd.grad(self.loss(y_inv), y_forward)[0]

            # Now compute the same gradients, but using the implicit differentiation layer with a custom backward pass
            y_inv_implicit = implicit_inverse_layer(y_forward, inv, forward_jacobian=forward_jacobian)
            gradients_implicit = torch.autograd.grad(self.loss(y_inv_implicit), y_forward)[0]
            self.assertTrue(torch.allclose(gradients_gt, gradients_implicit, atol=self.atol, rtol=self.rtol))

