from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext


setup(
    name='QuickMeanShiftPP',
    version='1.0',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("QuickMeanShiftPP",
                 sources=["quickmeanshiftpp.pyx"],
                 language="c++",
                 include_dirs=[numpy.get_include()])],
    author='paper id: 5122',
    author_email='paper id: 5122'

)
