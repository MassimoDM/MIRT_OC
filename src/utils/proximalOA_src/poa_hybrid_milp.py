# @Author: Massimo De Mauri <massimo>
# @Date:   2019-03-07T17:07:54+01:00
# @Email:  massimo.demauri@gmail.com
# @Filename: poa_hybrid_milp.py
# @Last modified by:   massimo
# @Last modified time: 2019-03-07T17:30:20+01:00
# @License: apache 2.0

import MIOCtoolbox as oc

def setup_milp_hyb(lbx,ubx,dsc_i,matrices,memory,options):

    # default options
    opts = {}
    opts['solver'] = 'openbb'
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

    for k in range(n_dsc):
        As[k,dsc_i[k]] = 1
        As[k,n_var+k] = -1
        As[n_dsc+k,dsc_i[k]] = -1
        As[n_dsc+k,n_var+k] = -1

    # create the constraints
    A = oc.vertcat(oc.horzcat(oc.sparsify(matrices[0]),oc.DM(matrices[0].shape[0],len(dsc_i))),As)
    l = oc.vertcat(matrices[1],ls)
    u = oc.vertcat(matrices[2],us)

    # initial values and bounds
    lbx = oc.vertcat(lbx,oc.DM(n_dsc,1))
    ubx = oc.vertcat(ubx,ubx[dsc_i]-lbx[dsc_i])

    # objective
    obj = oc.horzcat(1,oc.DM(1,n_var-1))




    memory = {'options':opts}
    return memory


def poa_milp_hyb(x0,lbx,ubx,dsc_i,matrices,alpha,obj_gain,memory,options):


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

    for k in range(n_dsc):
        As[k,dsc_i[k]] = 1
        As[k,n_var+k] = -1
        As[n_dsc+k,dsc_i[k]] = -1
        As[n_dsc+k,n_var+k] = -1

    # create the constraints
    A = oc.vertcat(oc.horzcat(oc.sparsify(matrices[0]),oc.DM(matrices[0].shape[0],len(dsc_i))),As)
    l = oc.vertcat(matrices[1],ls)
    u = oc.vertcat(matrices[2],us)

    # initial values and bounds
    x0 =  oc.vertcat(x0,oc.DM(n_dsc,1))
    lbx = oc.vertcat(lbx,oc.DM(n_dsc,1))
    ubx = oc.vertcat(ubx,ubx[dsc_i]-lbx[dsc_i])

    # objective
    obj = oc.horzcat(alpha*obj_gain,oc.DM(1,n_var-1),(1-alpha)*oc.DM.ones(1,n_dsc))


    if opts['solver'] == 'cplex':
        dsc = [False]*(n_var+n_dsc)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['discrete'] = dsc
        solver_opts['cplex.CPX_PARAM_EPRHS'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['cplex.CPX_PARAM_TILIM'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['cplex.CPX_PARAM_EPGAP'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['cplex.CPX_PARAM_EPAGAP'] = opts['MIPGapAbs']
        if opts['hybrid_mode']: solver_opts['cplex.CPX_PARAM_EPAGAP'] = oc.Inf

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
        return sol.call(arg)



    elif opts['solver'] == 'gurobi':

        dsc = [False]*(n_var+n_dsc)
        for k in dsc_i:
            dsc[k] = True

        # set the standard options
        solver_opts = {}
        solver_opts['discrete'] = dsc
        solver_opts['gurobi.FeasibilityTol'] = opts['primal_tol']
        if not opts['max_time'] is None: solver_opts['gurobi.TimeLimit'] = opts['max_time']
        if not opts['MIPGap'] is None: solver_opts['gurobi.MIPGap'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None: solver_opts['gurobi.MIPGapAbs'] = opts['MIPGapAbs']
        if opts['hybrid_mode']: solver_opts['gurobi.MIPGapAbs'] = oc.Inf

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
        return sol.call(arg)

    elif opts['solver'] == 'openbb':

        # identify the subsolver to use
        if not 'subsolver' in opts['other_opts']:
            subsolver = 'gurobi'
        else:
            subsolver = opts['other_opts'].pop('subsolver')

        # set standard opts
        solver_opts = {'verbose':False}
        solver_opts['primal_tol'] = opts['primal_tol']
        if not opts['max_time'] is None:
            solver_opts['time_limit'] = opts['max_time']
        if not opts['MIPGap'] is None:
            solver_opts['rel_gap_tol'] = opts['MIPGap']
        if not opts['MIPGapAbs'] is None:
            solver_opts['abs_gap_tol'] = opts['MIPGapAbs']

        # set possible additional options (from the user)
        for k in opts["other_opts"]:
            solver_opts[k] = opts["other_opts"][k]

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
        sol = oc.OpenBB(subsolver,arg,solver_opts)

        # solve the problem
        return sol.call(), memory

    else:
        raise NameError('ProximalOA, Solver unavailable')
