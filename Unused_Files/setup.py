import Cython.Build
import distutils.core

distutils.core.setup(ext_modules=Cython.Build.cythonize("calculate_bid_2.pyx"))
