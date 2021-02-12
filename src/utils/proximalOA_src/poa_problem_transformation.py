#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:44:05 2017

@author: Massimo De Mauri
"""

import MIOCtoolbox as oc

def poa_problem_transformation(problem,options=None):

    opts = {}
    opts['sos1'] = False

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('option not recognized'+k)


    # out = oc.binarize(problem,{'sos1':opts['sos1']})
    out = [{}]

    obj_v = oc.MX.sym('obj_v')
    problem.append_var_beg(obj_v,-oc.inf,oc.inf,0)
    problem.append_cns_beg(problem.obj['exp']-obj_v,-oc.inf,0)
    problem.obj['exp'] = obj_v

    x = oc.MX.sym('oval')
    out[0]['oval'] = oc.Function('vval',[x],[x])
    out[0]['vval'] = oc.Function('vval',[problem.var['sym']],[problem.var['sym'][1:]])
    out[0]['vlam'] = oc.Function('vlam',[problem.var['sym']],[problem.var['sym'][1:]])
    tmp = oc.MX.sym('clam',problem.cns['exp'].numel())
    out[0]['clam'] = oc.Function('clam',[tmp],[tmp[1:]])

    # out[0]['vval'] = oc.Function('vval',[problem.var['sym']],[out[0]['vval'](problem.var['sym'][1:])])
    # out[0]['vlam'] = oc.Function('vlam',[problem.var['sym']],[out[0]['vlam'](problem.var['sym'][1:])])
    # tmp = oc.MX.sym('clam',problem.cns['exp'].numel())
    # out[0]['clam'] = oc.Function('clam',[tmp],[out[0]['clam'](tmp[1:])])


    return out
