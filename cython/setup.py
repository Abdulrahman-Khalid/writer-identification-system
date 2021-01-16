from setuptools import setup
from Cython.Build import cythonize

setup(
    name='Write Identification System',
    ext_modules=cythonize("identify_writer.pyx"),
    zip_safe=False,
)