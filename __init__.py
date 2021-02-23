# @Author: Massimo De Mauri <massimo>
# @Date:   2020-12-30T12:20:17+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: __init__.py
# @Last modified by:   massimo
# @Last modified time: 2021-02-23T13:55:05+01:00
# @License: LGPL-3.0


import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8'); del locale

# import base components
from time import clock, sleep, time
from datetime import datetime
from warnings import warn
from os import getcwd,chdir,system, environ, path
from numpy import *
from casadi import *
from casadi.tools import capture_stdout, nice_stdout
from re import search as search_in_string
import importlib.util

# import some generic code
from .src.utils.casadiUtils                                         import *
from .src.utils.ParametricFunction                                  import *
from .src.utils.mathUtils                                           import *

# import main components
from .src.problem_formulation                                       import *
from .src.integration                                               import *
from .src.shooting                                                  import *
from .src.openbb_interface                                          import *
from .src.mpc                                                       import *


# openbb interface
spec = importlib.util.spec_from_file_location("openbb", environ['OPENBB_HOME']+"/alternative_interfaces/OpenBB.py")
openbb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(openbb)

# mpc addon
spec = importlib.util.spec_from_file_location("openbb", environ['OPENBB_HOME']+"/addons/MPCaddon/alternative_interfaces/OpenMPC.py")
mpc_addon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mpc_addon)

# home directory
home_dir = os.path.dirname(os.path.realpath(__file__))
