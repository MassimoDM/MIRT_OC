#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:24:56 2017

@author: Massimo De Mauri
"""

import MIOCtoolbox as oc


def poa_nlp_opt(problem,options = None):

    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-4
    opts['max_time'] = None
    opts['print_time'] = False
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4



    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp_opt - option not recognized: ' + k)

    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1




    # parametrize the problem
    x0 = oc.MX.sym('x0',problem.var['sym'].numel())
    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel())
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel())
    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())



    # define the problem
    prob = {'f':problem.obj['exp'],\
            'x':problem.var['sym'],\
            'p':oc.vertcat(x0,lbx,ubx,lmx0,lmg0),\
            'g':problem.cns['exp']}


    # general options
    options = {}
    options['print_time'] = opts['print_time']
    ftol = opts['primal_tol']
    otol = opts['primal_tol']*1e-2
    ctol = opts['primal_tol']*1e-2

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})


    if opts['solver'] == 'ipopt':

        options['ipopt.print_level'] = opts['print_level']

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.linear_solver'] = 'ma27'
        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'
        options['ipopt.expect_infeasible_problem'] = 'no'
        options['ipopt.required_infeasibility_reduction'] = 1e-2

        options['ipopt.mu_strategy'] = 'adaptive'
        options['ipopt.mu_oracle'] = 'probing'
        options['ipopt.gamma_phi'] = 1e-8
        options['ipopt.gamma_theta'] = 1e-4

        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 30
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75

        if not opts['max_iter'] is None: options['ipopt.max_iter'] = opts['solver_opts']['max_iter']
        if not opts['max_time'] is None: options['ipopt.max_cpu_time'] = opts['max_time']

        sol = oc.nlpsol('solver_name','ipopt', prob,options)

    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.RelaxCon'] = False


        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol
        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = otol

        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1

        options['worhp.Crossover'] = 2
        options['worhp.CrossoverTol'] = opts['primal_tol']**2

        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        raise NameError('Not implemented yet')

    arg = {'x0':  x0,\
           'p':   oc.vertcat(x0,lbx,ubx,lmx0,lmg0),\
           'lbx': lbx,\
           'ubx': ubx,\
           'lbg': problem.cns['lob'],\
           'ubg': problem.cns['upb'],\
           'lam_x0': lmx0,\
           'lam_g0': lmg0}

    out = sol.call(arg)

    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,lmx0,lmg0],\
                       [out['f'],out['x'],out['lam_x'],out['lam_g']],\
                       ['x0','lob','upb','lmx0','lmg0'],['obj','var','lmx','lmg'])

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def poa_nlp_L2(problem,options):
    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['print_time'] = False
    opts['primal_tol'] = 1e-8
    opts['max_time'] = None
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp - option not recognized: ' + k)


    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1




    dsc_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 0]

    x0 = oc.MX.sym('x0',problem.var['sym'].numel())
    dsc_ass = oc.MX.sym('dsc_ass',len(dsc_i))

    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel())
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel())

    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())


    # search for the feasible point closest to the discrete assignment dsc_ass
    dist = problem.var['sym'][dsc_i]-dsc_ass
    prob = {'f':(dist.T@dist)/dist.numel(),\
            'x':problem.var['sym'],\
            'p':oc.vertcat(dsc_ass,x0,lmx0,lmg0,lbx,ubx),\
            'g':problem.cns['exp']}


    # general options
    options = {}
    options['print_time'] = opts['print_time']

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})

    # tolerances
    ftol = opts['primal_tol']
    otol = 1e-4*opts['primal_tol']**2
    ctol = 1e-4*opts['primal_tol']**2


    if opts['solver'] == 'ipopt':

        options['ipopt.print_level'] = opts['print_level']
        options['ipopt.linear_solver'] = 'ma27'

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'
        options['ipopt.expect_infeasible_problem'] = 'no'
        options['ipopt.required_infeasibility_reduction'] = 1e-2

        options['ipopt.mu_strategy'] = 'adaptive'
        options['ipopt.mu_oracle'] = 'probing'
        options['ipopt.gamma_phi'] = 1e-8
        options['ipopt.gamma_theta'] = 1e-4
        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 0
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75

        if not opts['max_iter'] is None: options['ipopt.max_iter'] = opts['solver_opts']['max_iter']
        if not opts['max_time'] is None: options['ipopt.max_cpu_time'] = opts['max_time']

        sol = oc.nlpsol('solver_name','ipopt', prob,options)


    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol

        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = 100*otol


        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1


        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        print(opts['solver'])

        raise NameError('Not implemented yet')

    arg = {'x0':  x0,\
           'p':   oc.vertcat(dsc_ass,x0,lmx0,lmg0,lbx,ubx),\
           'lbx': lbx,\
           'ubx': ubx,\
           'lbg': problem.cns['lob'],\
           'ubg': problem.cns['upb'],\
           'lam_x0': lmx0,\
           'lam_g0': lmg0}

    out = sol.call(arg)





    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,dsc_ass,lmx0,lmg0],\
                       [out['f'],out['x'],out['lam_x'],out['lam_g']],\
                       ['x0','lob','upb','dsc_ass','lmx0','lmg0'],['obj','var','lmx','lmg'])


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def poa_nlp_Linf(problem,options):
    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-8
    opts['max_time'] = None
    opts['print_time'] = False
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp - option not recognized: ' + k)

    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1

    dsc_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 0]

    x0 = oc.MX.sym('x0',problem.var['sym'].numel())
    dsc_ass = oc.MX.sym('dsc_ass',len(dsc_i))

    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel())
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel())

    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())


    # if mode == 0 search for the feasible point closest to the discrete assignment dsc_ass
    # else fix the discrete assignment and try to optimize
    tmp = oc.MX(problem.var['sym'].numel(),1)
    tmp[cnt_i] = problem.var['sym'][cnt_i]
    tmp[dsc_i] = dsc_ass


    dist = problem.var['sym'][dsc_i]-dsc_ass


    obj = oc.MX.sym('obj')


    cnss = [oc.vertcat(dist-obj,-dist-obj),-oc.DM.inf(2*len(dsc_i)),oc.DM.zeros(2*len(dsc_i))]


    prob = {'f':obj,\
            'x':oc.vertcat(obj,problem.var['sym']),\
            'p':oc.vertcat(dsc_ass,lbx,ubx),\
            'g':oc.vertcat(cnss[0],problem.cns['exp'])}


    # general options
    options = {}
    options['print_time'] = opts['print_time']

    ftol = opts['primal_tol']
    otol = 1e-2*opts['primal_tol']
    ctol = 1e-2*opts['primal_tol']

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})



    if opts['solver'] == 'ipopt':


        options['ipopt.print_level'] = opts['print_level']
        options['ipopt.linear_solver'] = 'ma27'

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'

        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 0
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75


        if not opts['max_iter'] is None:
            options['ipopt.max_iter'] = opts['solver_opts']['max_iter']


        if not opts['max_time'] is None:
            options['ipopt.max_cpu_time'] = opts['max_time']


        sol = oc.nlpsol('solver_name','ipopt', prob,options)


    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol
        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = 100*otol
        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1



        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        print(opts['solver'])

        raise NameError('Not implemented yet')




    arg = {'x0':  oc.vertcat(0,x0),\
           'p':   oc.vertcat(dsc_ass,lbx,ubx),\
           'lbx': oc.vertcat(0,lbx),\
           'ubx': oc.vertcat(oc.DM.inf(),ubx),\
           'lbg': oc.vertcat(cnss[1],problem.cns['lob']),\
           'ubg': oc.vertcat(cnss[2],problem.cns['upb']),\
           'lam_x0': oc.vertcat(0,lmx0),\
           'lam_g0': oc.vertcat(oc.DM.zeros(2*len(dsc_i)),lmg0)}

    out = sol.call(arg)





    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,dsc_ass,lmx0,lmg0],\
                       [out['f'],out['x'][1:],out['lam_x'][1:],out['lam_g'][2*len(dsc_i):]],\
                       ['x0','lob','upb','dsc_ass','lmx0','lmg0'],['obj','var','lmx','lmg'])

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


def poa_nlp_L1(problem,options):
    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-8
    opts['max_time'] = None
    opts['print_time'] = False
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp - option not recognized: ' + k)

    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1


    dsc_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 0]

    x0 = oc.MX.sym('x0',problem.var['sym'].numel())
    dsc_ass = oc.MX.sym('dsc_ass',len(dsc_i))

    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel())
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel())

    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())


    # if mode == 0 search for the feasible point closest to the discrete assignment dsc_ass
    # else fix the discrete assignment and try to optimize
    tmp = oc.MX(problem.var['sym'].numel(),1)
    tmp[cnt_i] = problem.var['sym'][cnt_i]
    tmp[dsc_i] = dsc_ass

    dist = problem.var['sym'][dsc_i]-dsc_ass

    slack = oc.MX.sym('slack',len(dsc_i))

    cnss = [oc.vertcat(dist-slack,-dist-slack),-oc.DM.inf(2*len(dsc_i)),oc.DM.zeros(2*len(dsc_i))]

    prob = {'f':oc.sum1(slack)/slack.numel(),\
            'x':oc.vertcat(slack,problem.var['sym']),\
            'p':oc.vertcat(dsc_ass,lbx,ubx),\
            'g':oc.vertcat(cnss[0],problem.cns['exp'])}


    # general options
    options = {}
    options['print_time'] = opts['print_time']

    ftol = opts['primal_tol']
    otol = 1e-2*opts['primal_tol']
    ctol = 1e-2*opts['primal_tol']

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})



    if opts['solver'] == 'ipopt':


        options['ipopt.print_level'] = opts['print_level']
        options['ipopt.linear_solver'] = 'ma27'

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'

        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 0
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75


        if not opts['max_iter'] is None:
            options['ipopt.max_iter'] = opts['solver_opts']['max_iter']


        if not opts['max_time'] is None:
            options['ipopt.max_cpu_time'] = opts['max_time']


        sol = oc.nlpsol('solver_name','ipopt', prob,options)


    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol

        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = 100*otol




        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1



        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        print(opts['solver'])

        raise NameError('Not implemented yet')




    arg = {'x0':  oc.vertcat(oc.DM.zeros(len(dsc_i)),x0),\
           'p':   oc.vertcat(dsc_ass,lbx,ubx),\
           'lbx': oc.vertcat(oc.DM.zeros(len(dsc_i)),lbx),\
           'ubx': oc.vertcat(oc.DM.inf(len(dsc_i)),ubx),\
           'lbg': oc.vertcat(cnss[1],problem.cns['lob']),\
           'ubg': oc.vertcat(cnss[2],problem.cns['upb']),\
           'lam_x0': oc.vertcat(oc.DM.zeros(len(dsc_i)),lmx0),\
           'lam_g0': oc.vertcat(oc.DM.zeros(2*len(dsc_i)),lmg0)}

    out = sol.call(arg)





    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,dsc_ass,lmx0,lmg0],\
                       [out['f'],out['x'][len(dsc_i):],out['lam_x'][len(dsc_i):],out['lam_g'][2*len(dsc_i):]],\
                       ['x0','lob','upb','dsc_ass','lmx0','lmg0'],['obj','var','lmx','lmg'])








################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def poa_nlp(problem,options):


    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-8
    opts['max_time'] = None
    opts['print_time'] = False
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp - option not recognized: ' + k)

    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1


    dsc_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 0]

    x0 = oc.MX.sym('x0',problem.var['sym'].numel()+len(dsc_i))
    dsc_ass = oc.MX.sym('dsc_ass',len(dsc_i))

    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel()+len(dsc_i))
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel()+2*len(dsc_i))

    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())


    w = oc.MX.sym('w',3)
    fx_dsc = oc.MX.sym('fx_dsc')




    tmp = oc.MX(problem.var['sym'].numel(),1)
    tmp[cnt_i] = problem.var['sym'][cnt_i]
    tmp[dsc_i] = dsc_ass


    dist = problem.var['sym'][dsc_i]-dsc_ass

    L1_slack = oc.MX.sym('slack',len(dsc_i))

    L1_cnss = [oc.vertcat(dist-L1_slack,-dist-L1_slack),-oc.DM.inf(2*len(dsc_i)),oc.DM.zeros(2*len(dsc_i))]


    # three possible objective functions
    L1 = oc.sum1(L1_slack)
    L2 = dist.T@dist
    f = problem.obj['exp']





    prob = {'f':w[0]*f + w[1]*L2 + w[2]*L1,\
            'x':oc.vertcat(problem.var['sym'],L1_slack),\
            'p':oc.vertcat(w,dsc_ass,lbx,ubx),\
            'g':oc.vertcat(problem.cns['exp'],L1_cnss[0])}


    # general options
    options = {}
    options['print_time'] = opts['print_time']

    ftol = opts['primal_tol']
    otol = opts['primal_tol']*opts['integer_gap']
    ctol = 1e-2*opts['primal_tol']**2

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})


    if opts['solver'] == 'ipopt':

        options['ipopt.print_level'] = opts['print_level']
        options['ipopt.linear_solver'] = 'ma27'

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'

        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 0
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75

        options['ipopt.mu_strategy'] = 'adaptive'
        options['ipopt.mu_oracle'] = 'probing'

        options['ipopt.gamma_phi'] = 1e-8
        options['ipopt.gamma_theta'] = 1e-4


        if not opts['max_iter'] is None:
            options['ipopt.max_iter'] = opts['solver_opts']['max_iter']


        if not opts['max_time'] is None:
            options['ipopt.max_cpu_time'] = opts['max_time']


        sol = oc.nlpsol('solver_name','ipopt', prob,options)


    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol

        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = 100*otol




        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1



        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        print(opts['solver'])

        raise NameError('Not implemented yet')




    arg = {'x0':  x0,\
           'p':   oc.vertcat(w,dsc_ass,lbx,ubx),\
           'lbx': oc.vertcat(lbx,oc.DM.zeros(len(dsc_i))),\
           'ubx': oc.vertcat(ubx,ubx[dsc_i]-lbx[dsc_i]),\
           'lbg': oc.vertcat(problem.cns['lob'],L1_cnss[1]),\
           'ubg': oc.vertcat(problem.cns['upb'],L1_cnss[2]),\
           'lam_x0': oc.vertcat(lmx0),\
           'lam_g0': oc.vertcat(lmg0)}


    out = sol.call(arg)





    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,dsc_ass,lmx0,lmg0,w],\
                       [out['f'],out['x'],out['lam_x'],out['lam_g']],\
                       ['x0','lob','upb','dsc_ass','lmx0','lmg0','w'],\
                       ['obj','var','lmx','lmg'])
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def poa_nlp_simple(problem,options):


    opts = {}
    opts['solver'] = 'ipopt'
    opts['max_iter'] = None
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-8
    opts['max_time'] = None
    opts['print_time'] = False
    opts['code_gen'] = False
    opts['integer_gap'] = 1e-4

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('poa_nlp - option not recognized: ' + k)

    # printing options
    if opts['print_level'] >0:
        opts['print_time'] = True
        opts['print_level'] += -1


    dsc_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(len(problem.var['dsc'])) if problem.var['dsc'][i] == 0]

    x0 = oc.MX.sym('x0',problem.var['sym'].numel()+len(dsc_i))
    dsc_ass = oc.MX.sym('dsc_ass',len(dsc_i))

    lmx0 = oc.MX.sym('lx',problem.var['sym'].numel()+len(dsc_i))
    lmg0 = oc.MX.sym('lg',problem.cns['exp'].numel()+2*len(dsc_i))

    lbx = oc.MX.sym('lbx',problem.var['sym'].numel())
    ubx = oc.MX.sym('ubx',problem.var['sym'].numel())


    w = oc.MX.sym('w',2)
    tmp = oc.MX(problem.var['sym'].numel(),1)
    tmp[cnt_i] = problem.var['sym'][cnt_i]
    tmp[dsc_i] = dsc_ass

    dist = problem.var['sym'][dsc_i]-dsc_ass

    # two possible objective functions
    L2 = dist.T@dist
    f = problem.obj['exp']


    prob = {'f':w[0]*f + w[1]*L2,\
            'x':oc.vertcat(problem.var['sym']),\
            'p':oc.vertcat(w,dsc_ass,lbx,ubx),\
            'g':oc.vertcat(problem.cns['exp'])}


    # general options
    options = {}
    options['print_time'] = opts['print_time']

    ftol = opts['primal_tol']
    otol = 1e-2*opts['primal_tol']
    ctol = 1e-2*opts['primal_tol']

    if not opts['code_gen'] is None and opts['code_gen']:
        options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})


    if opts['solver'] == 'ipopt':

        options['ipopt.print_level'] = opts['print_level']
        options['ipopt.linear_solver'] = 'ma27'

        options['ipopt.bound_relax_factor']= 0
        options['ipopt.honor_original_bounds'] = 'yes'

        options['ipopt.warm_start_init_point'] = 'yes'
        options['ipopt.fixed_variable_treatment'] = 'make_parameter'

        options['ipopt.constr_viol_tol'] = ftol
        options['ipopt.dual_inf_tol'] =  otol
        options['ipopt.compl_inf_tol'] =  ctol

        options['ipopt.acceptable_iter'] = 0
        options['ipopt.acceptable_constr_viol_tol'] = ftol
        options['ipopt.acceptable_dual_inf_tol'] = otol**.75
        options['ipopt.acceptable_compl_inf_tol'] = ctol**.75

        options['ipopt.mu_strategy'] = 'adaptive'
        options['ipopt.mu_oracle'] = 'probing'

        options['ipopt.gamma_phi'] = 1e-8
        options['ipopt.gamma_theta'] = 1e-4


        if not opts['max_iter'] is None:
            options['ipopt.max_iter'] = opts['solver_opts']['max_iter']


        if not opts['max_time'] is None:
            options['ipopt.max_cpu_time'] = opts['max_time']


        sol = oc.nlpsol('solver_name','ipopt', prob,options)


    elif opts['solver'] == 'worhp':

        options['worhp.NLPprint'] = max([-1,opts['print_level']-1])

        options['worhp.TolFeas'] = ftol
        options['worhp.TolOpti'] = otol
        options['worhp.TolComp'] = ctol

        options['worhp.AcceptTolFeas'] = ftol
        options['worhp.AcceptTolOpti'] = 100*otol




        options['worhp.UserDG'] = 1
        options['worhp.UserHM'] = 1



        if not opts['max_iter'] is None:
            options['worhp.MaxIter'] = opts['solver_opts']['max_iter']

        if not opts['max_time'] is None:
            options['worhp.Timeout'] = opts['max_time']

        sol = oc.nlpsol('solver_name','worhp', prob,options)
    else:
        print(opts['solver'])

        raise NameError('Not implemented yet')




    arg = {'x0':  x0,\
           'p':   oc.vertcat(w,dsc_ass,lbx,ubx),\
           'lbx': oc.vertcat(lbx,oc.DM.zeros(len(dsc_i))),\
           'ubx': oc.vertcat(ubx,ubx[dsc_i]-lbx[dsc_i]),\
           'lbg': oc.vertcat(problem.cns['lob'],L1_cnss[1]),\
           'ubg': oc.vertcat(problem.cns['upb'],L1_cnss[2]),\
           'lam_x0': oc.vertcat(lmx0),\
           'lam_g0': oc.vertcat(lmg0)}


    out = sol.call(arg)





    return oc.Function('FP_nlp',\
                       [x0,lbx,ubx,dsc_ass,lmx0,lmg0,w],\
                       [out['f'],out['x'],out['lam_x'],out['lam_g']],\
                       ['x0','lob','upb','dsc_ass','lmx0','lmg0','w'],\
                       ['obj','var','lmx','lmg'])
