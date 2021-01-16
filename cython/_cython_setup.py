from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='Write Identification System',
    ext_modules=cythonize("identify_writer.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)