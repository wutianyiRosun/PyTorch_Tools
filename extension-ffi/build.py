import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ['src/my_lib.c']
headers = ['src/my_lib.h']
defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['src/my_lib_cuda.c']
    headers += ['src/my_lib_cuda.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

#creates and configures a cffi.FFI object, that builds PyTorch extension
ffi = create_extension(
    '_ext.my_lib',  #name--package name. Can be a nested module e.g. "._ext.my_lib"
    headers=headers,  #header(str or List[str]--list of headers, that contain only exported functions
    sources=sources,  #sources(List[str])--list of sources to compile
    define_macros=defines,
    relative_to=__file__, #path of the build file. Required when "package is True". It's best to use "__file__" for this argument.,
    extra_compile_args=["-std=c99"]
)

if __name__ == '__main__':
    ffi.build()
