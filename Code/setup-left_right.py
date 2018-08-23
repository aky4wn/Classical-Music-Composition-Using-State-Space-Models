from distutils.core import setup
from Cython.Build import cythonize


setup(
  name = 'Baum Welch Code',
  ext_modules = cythonize("BaumWelchLR.pyx"),
)