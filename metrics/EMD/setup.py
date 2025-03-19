from setuptools import setup
from torch.tool.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })