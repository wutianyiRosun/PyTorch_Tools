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

## Steps:
(1) 编写c程序，实现前向计算和后向求导两个函数.如果利用cuda,则需实现对应cuda版本的程序

(2) 编写python 调用c的接口, Function或者Module. PyTorch中的tensor对应C代码中的THTensor

(3) 编写build.py编译文件

Note: 

Torch.utils.ffi: http://pytorch.org/docs/master/ffi.html

Extension API reference: https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension

cffi: http://cffi.readthedocs.io/en/latest/
