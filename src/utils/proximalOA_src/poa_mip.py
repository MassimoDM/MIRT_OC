# @Author: Massimo De Mauri <massimo>
# @Date:   2019-12-16T12:49:28+01:00
# @Email:  massimo.demauri@gmail.com
# @Filename: poa_mip.py
# @Last modified by:   massimo
# @Last modified time: 2020-04-09T11:32:24+02:00
# @License: LGPL-3.0


import MIOCtoolbox as oc

def poa_mi_L2(x0,lbx,ubx,dsc_i,matrices,alpha,norm_gain,options):


    opts = {}
    opts['solver'] = 'gurobi'
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-4
    opts['max_time'] = None
    opts['MIPGap'] = None
    opts['MIPGapAbs'] = None
    opts['integer_gap'] = 1e-4
    opts['other_opts'] = {}

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('Option not recognized: ' + k)


    # total number of variables
    n_var = matrices[0].shape[1]
    n_dsc = len(dsc_i)

    # create the objective
    Qd = oc.DM(n_var,1)
    Qd[dsc_i] = (1-alpha)*norm_gain
    Q = oc.diag(Qd)
    C = oc.horzcat(alpha,oc.DM(1,n_var-1))
    C[dsc_i] = -(1-alpha)*norm_gain*x0[dsc_i]

    # create the constraints
    A = oc.sparsify(matrices[0])
    l = matrices[1]
    u = matrices[2]

    if opts['solver'] == 'cplex':
        dsc = [False]*n_var
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['discrete'] = dsc
        solver_opts['cplex.CPXPARAM_MIP_Display'] = 0
        solver_opts['cplex.CPXPARAM_Simplex_Tolerances_Feasibility'] = opts['primal_tol']
        solver_opts['cplex.CPX_PARAM_EPINT'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['cplex.CPXPARAM_TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_AbsMIPGap'] = opts['MIPGapAbs']


        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['cplex.'+k] = opts["other_opts"][k]


        # constuct the solver
        sol = oc.conic('solver', 'cplex', {'a':A.sparsity(),'h':Q.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'h':Q,
               'g':C,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call(arg)
        return out



    elif opts['solver'] == 'gurobi':

        dsc = [False]*n_var
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['gurobi.OutputFlag'] = opts['print_level'] >0
        solver_opts['discrete'] = dsc
        solver_opts['gurobi.FeasibilityTol'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['gurobi.TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['gurobi.MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['gurobi.MIPGapAbs'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['gurobi.'+k] = opts["other_opts"][k]

        # construct the solver
        sol = oc.conic('solver', 'gurobi', {'a':A.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'h':Q,
               'g':C,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        return sol.call(arg)

    elif opts['solver'] == 'openbb':

        # set standard opts
        solver_opts = {'verbose':False}
        # identify the subsolver to use
        if not 'subsolver' in opts['other_opts']:
            solver_opts['subsolver'] = 'gurobi'
        else:
            solver_opts['subsolver'] = opts['other_opts'].pop('subsolver')

        solver_opts['primal_tol'] = opts['primal_tol']
        if not opts['max_time'] is None:
            solver_opts['time_limit'] = opts['max_time']
        if not opts['MIPGap'] is None:
            solver_opts['rel_gap_tol'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None:
            solver_opts['abs_gap_tol'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]: solver_opts[k] = opts["other_opts"][k]

        # define the problem
        arg = {'h':Q,
               'L':C,\
               'A':A,\
               'l':l,\
               'u':u,\
               'vl':lbx,\
               'vu':ubx,\
               'vv':x0,\
               'dsc_i':dsc_i}

        # construct the solver
        sol = oc.OpenBB(solver_opts.pop('subsolver'),arg,solver_opts)

        # solve the problem
        return sol.call()
    else:
        raise NameError('Not implemented yet')


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def poa_mi_L1(x0,lbx,ubx,dsc_i,matrices,alpha,norm_gain,options):


    opts = {}
    opts['solver'] = 'gurobi'
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-4
    opts['max_time'] = None
    opts['MIPGap'] = None
    opts['MIPGapAbs'] = None
    opts['integer_gap'] = 1e-4
    opts['other_opts'] = {}

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('Option not recognized: ' + k)


    # total number of variables
    n_var = matrices[0].shape[1]
    n_dsc = len(dsc_i)

    # create l1 norm constraints
    As = oc.DM(n_dsc*2,n_var+n_dsc)
    tmp = oc.DM.eye(n_dsc)

    As[:n_dsc,dsc_i]  =  tmp
    As[:n_dsc,n_var:] = -tmp

    As[n_dsc:,dsc_i]  = -tmp
    As[n_dsc:,n_var:] = -tmp

    ls = -oc.DM.ones(n_dsc*2)*oc.inf
    us =  oc.vertcat(x0[dsc_i],-x0[dsc_i])

    # create the constraints
    A = oc.vertcat(oc.horzcat(oc.sparsify(matrices[0]),oc.DM(matrices[0].shape[0],len(dsc_i))),As)
    l = oc.vertcat(matrices[1],ls)
    u = oc.vertcat(matrices[2],us)

    # initial values and bounds
    x0 =  oc.vertcat(x0,oc.DM(n_dsc,1))
    lbx = oc.vertcat(lbx,oc.DM(n_dsc,1))
    ubx = oc.vertcat(ubx,ubx[dsc_i]-lbx[dsc_i])

    # objective
    obj = oc.horzcat(alpha,oc.DM(1,n_var-1),(1-alpha)*norm_gain*oc.DM.ones(1,n_dsc))


    if opts['solver'] == 'cplex':
        dsc = [False]*(n_var+n_dsc)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['discrete'] = dsc
        solver_opts['cplex.CPXPARAM_MIP_Display'] = 0
        solver_opts['cplex.CPXPARAM_Simplex_Tolerances_Feasibility'] = opts['primal_tol']
        solver_opts['cplex.CPX_PARAM_EPINT'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['cplex.CPXPARAM_TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_AbsMIPGap'] = opts['MIPGapAbs']


        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['cplex.'+k] = opts["other_opts"][k]


        # constuct the solver
        sol = oc.conic('solver', 'cplex', {'a':A.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'g':obj,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call(arg)
                out['x'] = out['x'][:n_var]
        return out



    elif opts['solver'] == 'gurobi':

        dsc = [False]*(n_var+n_dsc)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['gurobi.OutputFlag'] = opts['print_level'] >0
        solver_opts['discrete'] = dsc
        solver_opts['gurobi.FeasibilityTol'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['gurobi.TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['gurobi.MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['gurobi.MIPGapAbs'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['gurobi.'+k] = opts["other_opts"][k]

        # construct the solver
        sol = oc.conic('solver', 'gurobi', {'a':A.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'g':obj,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call(arg)
                out['x'] = out['x'][:n_var]
        return out

    elif opts['solver'] == 'openbb':

        # set standard opts
        solver_opts = {'verbose':False}
        # identify the subsolver to use
        if not 'subsolver' in opts['other_opts']:
            solver_opts['subsolver'] = 'gurobi'
        else:
            solver_opts['subsolver'] = opts['other_opts'].pop('subsolver')

        solver_opts['primal_tol'] = opts['primal_tol']
        if not opts['max_time'] is None:
            solver_opts['time_limit'] = opts['max_time']
        if not opts['MIPGap'] is None:
            solver_opts['rel_gap_tol'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None:
            solver_opts['abs_gap_tol'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]: solver_opts[k] = opts["other_opts"][k]

        # define the problem
        arg = {'L':obj,\
               'A':A,\
               'l':l,\
               'u':u,\
               'vl':lbx,\
               'vu':ubx,\
               'vv':x0,\
               'dsc_i':dsc_i}

        # construct the solver
        sol = oc.OpenBB(solver_opts.pop('subsolver'),arg,solver_opts)

        # solve the problem
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call()
        return out
    else:
        raise NameError('Not implemented yet')



################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def poa_mi_Linf(x0,lbx,ubx,dsc_i,matrices,alpha,norm_gain,options):


    opts = {}
    opts['solver'] = 'gurobi'
    opts['print_level'] = 0
    opts['primal_tol'] = 1e-4
    opts['max_time'] = None
    opts['MIPGap'] = None
    opts['MIPGapAbs'] = None
    opts['integer_gap'] = 1e-4
    opts['other_opts'] = {}

    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('Option not recognized: ' + k)


    # total number of variables
    n_var = matrices[0].shape[1]
    n_dsc = len(dsc_i)

    # create l1 norm constraints
    As = oc.DM(n_dsc*2,n_var+1)
    tmp = oc.DM.eye(n_dsc)

    As[:n_dsc,dsc_i]  =  tmp
    As[:n_dsc,n_var:] = -1

    As[n_dsc:,dsc_i]  = -tmp
    As[n_dsc:,n_var:] = -1

    ls = -oc.DM.ones(n_dsc*2)*oc.inf
    us =  oc.vertcat(x0[dsc_i],-x0[dsc_i])


    # create the constraints
    A = oc.vertcat(oc.horzcat(oc.sparsify(matrices[0]),oc.DM(matrices[0].shape[0],1)),As)
    l = oc.vertcat(matrices[1],ls)
    u = oc.vertcat(matrices[2],us)

    # initial values and bounds
    x0 =  oc.vertcat(x0,0)
    lbx = oc.vertcat(lbx,0)
    ubx = oc.vertcat(ubx,oc.inf)

    # objective
    obj = oc.horzcat(alpha,oc.DM(1,n_var-1),(1-alpha)*norm_gain)


    if opts['solver'] == 'cplex':
        dsc = [False]*(n_var+1)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['discrete'] = dsc
        solver_opts['cplex.CPXPARAM_MIP_Display'] = 0
        solver_opts['cplex.CPXPARAM_Simplex_Tolerances_Feasibility'] = opts['primal_tol']
        solver_opts['cplex.CPX_PARAM_EPINT'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['cplex.CPXPARAM_TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['cplex.CPXPARAM_MIP_Tolerances_AbsMIPGap'] = opts['MIPGapAbs']


        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['cplex.'+k] = opts["other_opts"][k]


        # constuct the solver
        sol = oc.conic('solver', 'cplex', {'a':A.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'g':obj,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call(arg)
                out['x'] = out['x'][:n_var]
        return out



    elif opts['solver'] == 'gurobi':

        dsc = [False]*(n_var+1)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['gurobi.OutputFlag'] = opts['print_level'] >0
        solver_opts['discrete'] = dsc
        solver_opts['gurobi.FeasibilityTol'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['gurobi.TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['gurobi.MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['gurobi.MIPGapAbs'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts['gurobi.'+k] = opts["other_opts"][k]

        # construct the solver
        sol = oc.conic('solver', 'gurobi', {'a':A.sparsity()},solver_opts)

        # define the problem
        arg = {'x0': x0,\
               'lbx': lbx,\
               'ubx': ubx,\
               'g':obj,\
               'a':A,\
               'lba': l,\
               'uba': u}

        # solve
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call()
        return out

    elif opts['solver'] == 'openbb':

        # set standard opts
        solver_opts = {'verbose':False}
        # identify the subsolver to use
        if not 'subsolver' in opts['other_opts']:
            solver_opts['subsolver'] = 'cplex'
        else:
            solver_opts['subsolver'] = opts['other_opts'].pop('subsolver')

        solver_opts['primal_tol'] = opts['primal_tol']
        if not opts['max_time'] is None:
            solver_opts['time_limit'] = opts['max_time']
        if not opts['MIPGap'] is None:
            solver_opts['rel_gap_tol'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None:
            solver_opts['abs_gap_tol'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]: solver_opts[k] = opts["other_opts"][k]

        # define the problem
        arg = {'L':obj,\
               'A':A,\
               'l':l,\
               'u':u,\
               'vl':lbx,\
               'vu':ubx,\
               'vv':x0,\
               'dsc_i':dsc_i}

        # construct the solver
        sol = oc.OpenBB(solver_opts.pop('subsolver'),arg,solver_opts)

        # solve the problem
        with oc.tools.capture_stdout() as solver_out:
            with oc.tools.nice_stdout():
                out = sol.call()
        return out
    else:
        raise NameError('Not implemented yet')
