# README.md

Assume you have two functions: $f(y)$ and $g(x)$ such that $f(g(x)) = x$, i.e., $g$ is the right inverse to $f$.
The [inverse function theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem) allows us to compute the Jacobian and/or gradient of $g$.
This is useful when $g(x)$ is a function you can't easily explicitly write down (it's no coincidence that the [implicit function theorem}(https://en.wikipedia.org/wiki/Implicit_function_theorem) can be used to proof the inverse function theorem and vice versa).
It's also useful when you have an implementation of $g(x)$ that you don't want to or can't implement in torch.

Anyways, you end up wanting to compute the gradient of $g(x)$ and you can't do so using autograd.
As long as you have a differentiable version of it's left inverse $f(x)$, the functionalities provided in this repository can help you out!
It implements a custom functional that wraps $g(x)$ and computes its gradient following the inverse function theorem.


## Differentiate a function by differentiating its inverse!

This repository contains two Python files:

1. `inverse_differentiation_layer.py`
2. `test_implicit_differentiation.py`

### `inverse_differentiation_layer.py`

This file contains an implementation to different. It includes a class `ImplicitInverseLayer` that extends PyTorch's `Function` class. This class is used to implement a custom backward pass for $g(x)§ which is the inverse of some $f(y)$. The gradients for the backward pass are computed using the inverse function theorem.

The file also contains a function `batch_jacobian` that computes the Jacobian of a function with respect to its inputs, assuming that the first dimension is a batch dimension. This function is used to compute the Jacobian of the forward function in the `ImplicitInverseLayer` class.

Finally, the file contains a function `implicit_inverse_layer` that is a wrapper around the `ImplicitInverseLayer` class, allowing for a more convenient usage.

### `test_implicit_differentiation.py`

This file contains tests for the `ImplicitInverseLayer` class and the `batch_jacobian` function implemented in `custom_gradients.py`. The tests ensure that the custom gradients and the Jacobian computation are working correctly.

## Installation

To use the code in this repository, you need to have Python and PyTorch installed. You can install PyTorch using pip:

```bash
pip install torch
```

After that, just use the `implicit_inverse_layer` function as you would use any other torch function.

## Testing

To run the tests in `test_implicit_differentiation.py`, you can use a test runner like pytest:

```bash
pytest test_implicit_differentiation.py
```

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.
