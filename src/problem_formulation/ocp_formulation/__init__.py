from .constraint import *
from .input import *
from .oc_problem import *
from .variable import *
from .sos1_constraint import *

# some wrapper
def  eq(lhs,rhs=0,lbl=''): return constraint(lhs - rhs,0,0,'equality',None,lbl)
def leq(lhs,rhs=0,lbl=''): return constraint(lhs-rhs,-oc.inf,0,'inequality',None,lbl)    
def geq(lhs,rhs=0,lbl=''): return constraint(rhs-lhs,-oc.inf,0,'inequality',None,lbl)     