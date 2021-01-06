#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:14:34 2017

@author: Massimo De Mauri
"""

from time import time
from warnings import warn
import casadi as cs
from casadi.tools import capture_stdout, nice_stdout
from re import search as search_in_string

def solve_with_bonmin(problem,options=None):


    start_t = time()

    # default options
    opts = {}
    opts['algorithm'] = 'BB'
    opts['linear_solver'] = 'ma27'
    opts['mi_solver_name'] = 'Cplex'
    opts['nl_solver_name'] = 'Ipopt'
    opts['add_SOS1'] = False
    opts['print_level'] = 0
    opts['max_time'] = None
    opts['max_iter'] = None
    opts['extra_opts'] = None
    opts['MIPGap'] = None
    opts['MIPGapAbs'] = None
    opts['primal_tol'] = 0.0001
    opts['code_gen'] = False

    # user defined options
    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                warn('solve_with_bonmin - option not recognized: ' + k)




    sol_opts = {}
    sol_opts['discrete'] = problem.var['dsc']
    sol_opts['bonmin.print_level'] = min(opts['print_level'],1)
    sol_opts['bonmin.linear_solver'] = 'ma86'
    if opts['mi_solver_name'] in ['cplex','Cplex'] :
        sol_opts['bonmin.milp_solver'] = opts['mi_solver_name']
    elif opts['mi_solver_name'] in ['cbc','Cbc','Cbc_D']:
        sol_opts['bonmin.milp_solver'] = 'Cbc_D'
    elif opts['mi_solver_name'] in ['cbc_par','Cbc_Par']:
        sol_opts['bonmin.milp_solver'] = 'Cbc_Par'
    else:
        raise NameError('MILP subsolver unavailable')

    sol_opts['bonmin.nlp_solver'] = opts['nl_solver_name']
    sol_opts['bonmin.algorithm'] = 'B-'+opts['algorithm']
    sol_opts['bonmin.cutoff_decr'] = opts['primal_tol']
    sol_opts['bonmin.integer_tolerance'] = opts['primal_tol']
    sol_opts['bonmin.constr_viol_tol'] = opts['primal_tol']
    sol_opts['bonmin.acceptable_dual_inf_tol'] = opts['primal_tol']
    if not opts['max_iter'] is None: sol_opts['bonmin.max_iter'] = opts['max_iter']
    # sol_opts['bonmin.milp_sub.tolerances_integrality'] = opts['primal_tol']
    # sol_opts['bonmin.milp_sub.simplex.tolerance.feasibility'] = opts['primal_tol']



    if not opts['extra_opts'] is None:
        for k in opts['extra_opts']:
            sol_opts['bonmin.'+k] = opts['extra_opts'][k]

    if not opts['max_time'] is None:
        sol_opts['bonmin.time_limit'] = opts['max_time']

    if not opts['MIPGap'] is None:
        sol_opts['bonmin.allowable_fraction_gap'] = opts['MIPGap']

    if not opts['MIPGapAbs'] is None:
        sol_opts['bonmin.allowable_gap'] = opts['MIPGapAbs']

    if not problem.sos1 is None and opts['add_SOS1']:
        sos_groups = [gl for s in problem.sos1 for gl in s['g'] ]
        if len(sos_groups):
            sol_opts['sos_groups'] = sos_groups

    if opts['code_gen']:
        sol_opts.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})



    prob = {'f':problem.obj['exp'],'x':problem.var['sym']}
    if not problem.cns['exp'] is None: prob['g'] = problem.cns['exp']



    sol = cs.nlpsol('solver_name','bonmin', prob,sol_opts)

    arg = {}
    arg['x0'] =  problem.var['val']
    arg['lbx'] = problem.var['lob']
    arg['ubx'] = problem.var['upb']

    if not problem.cns['exp'] is None:
        arg['lbg'] = problem.cns['lob']
        arg['ubg'] = problem.cns['upb']

    start_time_ = time()
    with capture_stdout() as solver_out:
        with nice_stdout():
            out = sol.call(arg)
    solution_time = time()-start_time_

    # scan the solver output for information
    lines = solver_out[0].split('\n')
    iterations = 0
    for k in range(len(lines)):
        match = search_in_string(r'Performed [0-9]+', lines[k])
        if not match is None:
            iterations = int(match.group(0).split(' ')[1])
            break

    problem.var['val'] = out['x']
    problem.var['lam'] = out['lam_x']
    problem.cns['lam'] = out['lam_g']


    stats = sol.stats()
    stats['total_time'] = time()-start_t
    stats['solution_time'] = solution_time
    stats['max_violation'] = problem.get_max_violation()
    stats['num_iterations'] = iterations

    if max([stats['max_violation'] ['bounds'],stats['max_violation'] ['constraints']]) < opts['primal_tol']:
        stats['return_status'] = 'Success'
    else:
        stats['return_status'] = 'Failure'

    if opts['print_level'] >0:
        print(solver_out[0])
        print('\nReturn status:',stats['return_status'])
        print(' - Objective value:',out['f'])
        print(' - Max violation:',max([stats['max_violation'] ['bounds'],stats['max_violation'] ['constraints']]))
        print(' - Time spent:',stats['total_time'])
        print('\n')


    return stats
