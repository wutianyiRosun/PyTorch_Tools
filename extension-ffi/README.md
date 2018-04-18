# An example C extension for PyTorch

This example showcases adding a neural network layer that adds two input Tensors

- src: C source code
- functions: the autograd functions
- modules: code of the nn module
- build.py: a small file that compiles your module to be ready to use
- test_*.py: an example file that loads and uses the extension

```bash
python build.py
python test_function_version.py
python test_module_version.py
```

Note: 

Torch.utils.ffi: http://pytorch.org/docs/master/ffi.html

Extension API reference: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension

cffi: http://cffi.readthedocs.io/en/latest/
