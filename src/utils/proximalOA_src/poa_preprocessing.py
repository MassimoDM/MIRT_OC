#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 16:28:06 2017

@author: Massimo De Mauri
"""
import MIOCtoolbox as oc


def poa_preprocessing(problem,options=None):
    
    opts = {}
    opts['tolerance'] = 0.0001
    opts['lin_constraints'] = None
    opts['max_time'] = oc.inf
    
    if not options is None:
         for k in options:
             if k in opts:
                 opts[k] = options[k]
             else:
                 oc.warn('Option not recognized: ' + k)
                 
                 
    # the relaxed solution is a lower bound on the optimal discrete one
    problem.cns['exp'] = oc.vertcat(problem.cns['exp'],-problem.obj['exp']+problem.get_objective_value())
    problem.cns['lob'] = oc.vertcat(problem.cns['lob'],-oc.inf)
    problem.cns['upb'] = oc.vertcat(problem.cns['upb'],0)
    if not problem.cns['lam'] is None: problem.cns['lam'] = oc.vertcat(problem.cns['lam'],0)
    if not problem.cns['lbl'] is None: problem.cns['lbl'].append('<opt_lob>')
    
    clam_in = oc.MX.sym('clam',problem.cns['exp'].numel())
    oval_in = oc.MX.sym('oval')
    remap = {'vval':oc.Function('vval',[problem.var['sym']],[problem.var['sym']]),\
             'vlam':oc.Function('vlam',[problem.var['sym']],[problem.var['sym']]),\
             'clam':oc.Function('clam',[clam_in],[clam_in[:-1]]),\
             'oval':oc.Function('vval',[oval_in],[oval_in])}
    
    

    # use linear bound propagation to eliminate some discrete assignments
    if opts['lin_constraints'] is None:
        indxs = list(range(problem.cns['exp'].numel()))#problem.get_linear_cns()

        c_exp = problem.cns['exp'][indxs]
        c_lob = problem.cns['lob'][indxs]
        c_upb = problem.cns['upb'][indxs]
    
        # manipulate the constraints to get them into g(x)<b form                                   
        tmp1 = oc.nz_indices(oc.sparsify(c_lob >-oc.inf))[0] # finite lower bounds
        tmp2 = oc.nz_indices(oc.sparsify(c_upb < oc.inf))[0] # finite upper bounds
    
                        
        exp  = oc.vertcat(-c_exp[tmp1],c_exp[tmp2])
        b    = oc.vertcat(-c_lob[tmp1],c_upb[tmp2])
    
        # linearize the constraints around the current variables value
        JcnsF = oc.Function('Jcns',[problem.var['sym']],[oc.jacobian(exp,problem.var['sym']),exp])
    
        J = JcnsF(problem.var['val'])
        A = J[0]
        b = b - J[1] + J[0]@problem.var['val']
    
    else:
        
        A = opts['lin_constraints'][0]
        l = opts['lin_constraints'][1]
        u = opts['lin_constraints'][2]
        
        
        indxs1 = oc.nz_indices(oc.sparsify(l >-oc.inf))[0]
        indxs2 = oc.nz_indices(oc.sparsify(u < oc.inf))[0]
        
        A = oc.vertcat(A[indxs1,:],A[indxs2,:])
        b = oc.vertcat(l[indxs1],u[indxs2])



    out = oc.linear_bounds_propagation(A,b,oc.DM(problem.var['lob']),oc.DM(problem.var['upb']),{'dsc':problem.var['dsc'],'max_time':opts['max_time'],'tolerance':opts['tolerance']})
    
    dsc_i = [k for k in range(len(problem.var['dsc'])) if problem.var['dsc'][k]]
    
    problem.var['lob'][dsc_i] = out['lob'][dsc_i] 
    problem.var['upb'][dsc_i] = out['upb'][dsc_i]
    
    
    out = oc.elim_fixed_vars(problem,{'tolerance':opts['tolerance'],'print_level':1})

    out[0] = oc.remap_concat(remap,out[0])

    return out
