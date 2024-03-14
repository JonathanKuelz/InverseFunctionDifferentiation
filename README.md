Assume you have two functions: $f(y)$ and $g(x)$ such that $f(g(x)) = x$, i.e., $g$ is the right inverse to $f$.
The [inverse function theorem](https://en.wikipedia.org/wiki/Inverse_function_theorem) allows us to compute the Jacobian for any of them as long as we know the Jacobian for its counterpart.
This is useful when $g(x)$ is a function you can't quickly explicitly write down (it's no coincidence that the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) can be used to prove the inverse function theorem and vice versa).
It's also useful when you have an implementation of $g(x)$ that you don't want to or can't implement in torch.

## TL;DR
1. You have a function $g$ that you want to differentiate and autograd is not an option.
2. You do have a differentiable version of its inverse function $f$.
3. Then you can use the utilities provided in this repository to differentiate $g$.


## Differentiate a function by differentiating its inverse!

The classes `ImplicitInverseLayer` and `ImplicitInverseLayerAuto` extend PyTorch's `Function` class. They implements a custom backward pass with gradients computed using the inverse function theorem.
The function `implicit_inverse_layer` provides a wrapper around them, allowing for a more convenient usage.


## Example

```python
import torch
from inverse_differentiation_layer import implicit_inverse_layer

data = torch.rand((batch_size, 5), requires_grad=True)

def g(x):
  """g is the inverse to f, but you can't get the gradients with autograd"""
  with torch.no_grad():  # a more realistic example would be if this was calculated with another library, e.g., numpy
    return torch.asin(x)

y = implicit_inverse_layer(data, g, forward_func=torch.sin)
print(torch.autograd.grad(torch.sum(y), y)[0])  # Gradients are available, even though g cannot be differentiated with autograd
```

## Installation

To use the code in this repository, you must install Python and PyTorch. You can install PyTorch using pip:

```bash
pip install torch
```

After that, use the `implicit_inverse_layer` function as you would use any other torch function.

## Testing

To run the tests in `test_implicit_differentiation.py,` you can use a test runner like pytest:

```bash
pytest test_implicit_differentiation.py
```

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.
