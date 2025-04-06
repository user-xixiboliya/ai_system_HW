from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='myconv_cpp',
      ext_modules=[cpp_extension.CppExtension('myconv_cpp', ['myconv.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})