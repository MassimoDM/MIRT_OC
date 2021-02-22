# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-06T11:35:19+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: __init__.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-11T17:56:05+01:00
# @License: LGPL-3.0



from .constraint import *
from .input import *
from .OCmodel import *
from .variable import *
from .sos1_constraint import *

# some wrapper
def  eq(lhs,rhs=0,lbl=''): return constraint(lhs - rhs,0,0,'equality',None,lbl)
def leq(lhs,rhs=0,lbl=''): return constraint(lhs-rhs,-oc.inf,0,'inequality',None,lbl)
def geq(lhs,rhs=0,lbl=''): return constraint(rhs-lhs,-oc.inf,0,'inequality',None,lbl)
