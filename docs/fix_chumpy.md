You may need to modify `chumpy` package to avoid errors. 
   
  * Comment line 11 in `${Your_Conda_Environment}/lib/python3.11/site-packages/chumpy/__init__.py`:
  ```
  from .ch import *
  from .logic import *

  from .optimization import minimize
  from . import extras
  from . import testing
  from .version import version as __version__

  from .version import version as __version__

  # from numpy import bool, int, float, complex, object, unicode, str, nan, inf
  ```
  * Add *"inspect.getargspec = inspect.getfullargspec"* in `${Your_Conda_Environment}/lib/python3.11/site-packages/chumpy/ch.py` (line 25). Now it should look like:
  ```
  #!/usr/bin/env python
  # encoding: utf-8
  """
  Author(s): Matthew Loper

  See LICENCE.txt for licensing and contact information.
  """


  __all__ = ['Ch', 'depends_on', 'MatVecMult', 'ChHandle', 'ChLambda']

  import os, sys, time
  import inspect
  import scipy.sparse as sp
  import numpy as np
  import numbers
  import weakref
  import copy as external_copy
  from functools import wraps
  from scipy.sparse.linalg.interface import LinearOperator
  from .utils import row, col, timer, convert_inputs_to_sparse_if_necessary
  import collections
  from copy import deepcopy
  from functools import reduce
  inspect.getargspec = inspect.getfullargspec
  ```
