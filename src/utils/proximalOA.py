# @Author: Massimo De Mauri <massimo>
# @Date:   2019-12-16T15:38:48+01:00
# @Email:  massimo.demauri@gmail.com
# @Filename: proximalOA.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-14T18:58:43+01:00
# @License: LGPL-3.0


#TODO
# - re-implement hybrid mode

import MIOCtoolbox as oc
import os
import inspect
import sys
from multiprocessing import Process, Event, Pipe


# inspired by : A Feasibility Pump for mixed Integer Nonlinear Programs - Bonami et al.
def proximalOA(problem,options = None, nlps = None, dryrun = False):

    # default options container
    opts = {}
    # general options
    opts['max_iter'] = oc.DM.inf()
    opts['max_time'] = oc.DM.inf()
    opts['primal_tol'] = 1e-4
    opts['integer_tol'] = 1e-5
    opts['abs_opt_gap'] = 1e-6
    opts['rel_opt_gap'] = 1e-6
    opts['print_level'] = 1
    opts['initialized'] = False
    opts['cutoff'] = oc.inf
    opts['code_gen'] = True

    # Proximal OA specific options
    opts['algorithm'] = 1
    opts['alpha_min'] = 0.0
    opts['alpha_mem'] = 1/3
    opts['infeasibility_scale'] = 0.1

    # subsolvers options
    opts['nl_solver_name'] = 'ipopt'
    opts['nl_solver_norm'] = 'L2'
    opts['nl_solver_opts'] = {}
    opts['mi_solver_name'] = 'cplex'
    opts['mi_solver_norm'] = 'L1'
    opts['mi_solver_opts'] = {}

    # algotrithm modifiers
    opts['preprocessing_max_time'] = 0
    opts['limited_memory'] = False
    opts['bound_propagation'] = False
    opts['check_cycling'] = False
    opts['ignore_relaxation_failure'] = False
    opts['with_proximal_step'] = True


    if not options is None:
         for k in options:
             if k in opts and not opts[k] is None:
                 opts[k] = options[k]
             else:
                 oc.warn('proximalOA - option not recognized: ' + k)



    # check the alpha parameters for consistence
    assert(opts['alpha_mem'] >= 0 and opts['alpha_mem'] <= 1)

    # define MIP solver
    if opts['mi_solver_norm'] == 'L1':
        poa_mip = oc.poa_mi_L1
    elif opts['mi_solver_norm'] == 'L2':
        poa_mip = oc.poa_mi_L2
    elif opts['mi_solver_norm'] == 'Linf':
        poa_mip = oc.poa_mi_Linf
    else:
        raise NameError('Norm unavailable for POA MIP')

    # set solver options
    if opts['integer_tol'] is None:
        opts['integer_tol'] = opts['primal_tol']

    nlp_opts = {'solver':opts['nl_solver_name'],'primal_tol':opts['primal_tol'],
                'code_gen':opts['code_gen'],'print_level':max([0,opts['print_level']-3])}
    nlp_opts.update(opts['nl_solver_opts'])

    mip_opts = {'solver':opts['mi_solver_name'],'primal_tol':opts['primal_tol'],
                'print_level':max([0,opts['print_level']-3]),
                'integer_tol':opts['integer_tol']}
    mip_opts.update(opts['mi_solver_opts'])

    # set cutoff decrement
    opts['abs_opt_gap'] = max([opts['abs_opt_gap'],opts['primal_tol']])


    if opts['print_level'] > 0:
        print('\n=================================================================================')
        print('Proximal Outer Approximation:')
        print('     limited_memory =',opts['limited_memory'],'|','algorithm =',opts['algorithm'])
        print('     alpha parameters = min:',opts['alpha_min'],'mem:',opts['alpha_mem'],'scale:',opts['infeasibility_scale'])
        print('=================================================================================\n')



    initial_timestamp = oc.now()

    if opts['print_level'] > 1: print('\n+++ Preprocessing +++')
    elif opts['print_level'] > 0: print('- Preprocessing')

    # create a local copy of the problem to preprocess
    local_problem = problem.deepcopy()

    # transform the non-linear mixed-integer problem into a non-linear mixed-binary problem with linear objective
    pp_remap = oc.poa_problem_transformation(local_problem)[0]
    if opts['print_level'] > 2: print('- elapsed time:',oc.now() - initial_timestamp)
    local_problem.var['upb'][0] = opts['cutoff']


######################################################################################################

    # indices of the discrete and continuous variables
    dsc_i = [i for i in range(local_problem.var['sym'].numel()) if local_problem.var['dsc'][i] == 1]
    cnt_i = [i for i in range(local_problem.var['sym'].numel()) if local_problem.var['dsc'][i] == 0]

    # indices of the linear and non linear constraints
    lc_i  = local_problem.get_linear_cns()
    nlc_i = [i for i in range(local_problem.cns['exp'].numel()) if i not in lc_i]

    # get size information
    num_vars = len(dsc_i) + len(cnt_i)
    num_lc = len(lc_i)
    num_nlc = len(nlc_i)

    ######################################################################################################
    if opts['print_level'] > 1: print('\n+++ Building NLP Solvers and Non-Linear Functions +++')

    timestamp_ = oc.now()

    # get info on the linear constraints
    Flc = oc.Function('Flc',[local_problem.var['sym']],[local_problem.cns['exp'][lc_i]],['i'],['o'])
    Jlc = oc.Function('Jlc',[local_problem.var['sym']],[oc.jacobian(local_problem.cns['exp'][lc_i],local_problem.var['sym'])],['i'],['o'])
    Llc = local_problem.cns['lob'][lc_i]
    Ulc = local_problem.cns['upb'][lc_i]

    # get info on the non-linear constraints
    Fnlc = oc.Function('Fnlc',[local_problem.var['sym']],[local_problem.cns['exp'][nlc_i]],['i'],['o'])
    Jnlc = oc.Function('Jnlc',[local_problem.var['sym']],[oc.jacobian(local_problem.cns['exp'][nlc_i],local_problem.var['sym'])],['i'],['o'])
    Lnlc = local_problem.cns['lob'][nlc_i]
    Unlc = local_problem.cns['upb'][nlc_i]

    # code-generate non-linear functions
    if False and opts['code_gen']:

        current_directory = oc.getcwd()
        oc.chdir(oc.home_dir+'/jit_temp_files')

        # invoke the compiler
        CodeGen = oc.CodeGenerator('tmp_gen.c')
        if len(lc_i)  > 0: CodeGen.add(Flc); CodeGen.add(Jlc)
        if len(nlc_i) > 0: CodeGen.add(Fnlc); CodeGen.add(Jnlc)
        CodeGen.generate()
        oc.system('ccache gcc -fPIC -shared tmp_gen.c -o tmp_gen.so -O0')

        # overwrite the local functions to use their compiled versions
        Flc = oc.external('Flc', './tmp_gen.so')
        Jlc = oc.external('Jlc', './tmp_gen.so')
        Fnlc = oc.external('Fnlc', './tmp_gen.so')
        Jnlc = oc.external('Jnlc', './tmp_gen.so')

        oc.chdir(current_directory)

    # Build the non-linear solvers
    if nlps is None:
        nlp_opt = oc.poa_nlp_opt(local_problem,nlp_opts)
        if opts['nl_solver_norm'] == 'L2':
            nlp_prx = oc.poa_nlp_L2(local_problem,nlp_opts)
        elif opts['nl_solver_norm'] == 'L1':
            nlp_prx = oc.poa_nlp_L1(local_problem,nlp_opts)
        elif opts['nl_solver_norm'] == 'Linf':
            nlp_prx = oc.poa_nlp_Linf(local_problem,nlp_opts)
        elif not opts['nl_solver_norm'] is None:
            raise NameError
    else:
        nlp_opt = nlps[0]
        if not opts['nl_solver_norm'] is None:
            nlp_prx = nlps[1]

    compilation_time = oc.now() - timestamp_

    if dryrun == True:
        if not opts['nl_solver_norm'] is None:
            return (nlp_opt,nlp_prx)
        else:
            return (nlp_opt,)

    if opts['print_level'] > 2: print('- elapsed time:',oc.now() - initial_timestamp)
######################################################################################################

    if not opts['max_time'] is None and oc.now() - initial_timestamp >= opts['max_time']:
        if opts['print_level'] > 0:
            print('\n=================================================================================')
            print('Max time reached, Failure')
            print('- total time:',oc.now() - initial_timestamp)
            print('=================================================================================\n')

        elapsed_time =  oc.now() - initial_timestamp
        return {'return_status':'Failure','num_iterations':0,
                'total_time': elapsed_time,'solution_time':elapsed_time - compilation_time}



    if opts['print_level'] > 1:
        print('\n+++ Solving the relaxation +++')
        timestamp_ = oc.now()

    # find the relaxed optimal solution
    out = nlp_opt.call({'x0': local_problem.var['val'],\
                        'lob':local_problem.var['lob'],\
                        'upb':local_problem.var['upb'],\
                        'lmx0':oc.DM.zeros(local_problem.var['sym'].numel()),\
                        'lmg0':oc.DM.zeros(local_problem.cns['exp'].numel())})

    # store the optimal solution of the relaxed problem
    nlp_vval = oc.DM(out['var'])
    nlp_vlam = oc.DM(out['lmx'])
    nlp_clam = oc.DM(out['lmg'])

    # check the feasibility of the found solution
    Ft = oc.vertcat(Flc.call({'i':nlp_vval})['o'],Fnlc.call({'i':nlp_vval})['o'])
    Lt = oc.vertcat(Llc,Lnlc)
    Ut = oc.vertcat(Ulc,Unlc)


    if oc.dm_max(oc.vertcat(Lt-Ft,Ft-Ut)) < opts['primal_tol']:
        if opts['print_level'] > 1:
            print('- status: Success')
            print('- optimal objective value:',out['obj'])
            print(' - time spent:',oc.now() - timestamp_)

        # store the relaxed optimum
        rlxd_oval = oc.DM(out['obj'])
        rlxd_vval = oc.DM(nlp_vval)

    else:
        if opts['print_level'] > 1: print('- status: Failure')
        elapsed_time = oc.now() - initial_timestamp
        if not opts['ignore_relaxation_failure']:
            return {'return_status':'Failure','num_iterations':0,
                    'total_time': elapsed_time,'solution_time':elapsed_time - compilation_time}
    if opts['print_level'] > 2: print('- elapsed time:',oc.now() - initial_timestamp)





######################################################################################################
    if opts['print_level'] > 1: print('\n+++ Generating Linearizations +++\n')

    if num_lc > 0:
        # add the linear constraints into the linearizations set
        Ft = Flc.call({'i':nlp_vval})['o']
        Jt =  oc.sparsify(Jlc.call({'i':nlp_vval})['o'])

        A = Jt
        l = Llc - Ft + Jt@nlp_vval
        u = Ulc - Ft + Jt@nlp_vval
    else:
        A = oc.DM(0,num_vars)
        l = oc.DM()
        u = oc.DM()

    if num_nlc > 0:

        # initial set of linearizations: linearize problem around the relaxed optimum
        Ft = Fnlc.call({'i':nlp_vval})['o']
        Jt = oc.sparsify(Jnlc.call({'i':nlp_vval})['o'])

        if opts['limited_memory']:

            # get active constraints
            acti = oc.nz_indices(oc.sparsify((Ft<Lnlc+opts['primal_tol']) + (Ft>Unlc-opts['primal_tol'])))[0]

            # add the new linearizations to the new constraints set
            A = oc.vertcat(A,Jt[acti,:])
            l = oc.vertcat(l,Lnlc[acti] - Ft[acti] + Jt[acti,:]@nlp_vval)
            u = oc.vertcat(u,Unlc[acti] - Ft[acti] + Jt[acti,:]@nlp_vval)

        else:

            # add the new linearizations to the new constraints set
            A = oc.vertcat(A,Jt)
            l = oc.vertcat(l,Lnlc - Ft + Jt@nlp_vval)
            u = oc.vertcat(u,Unlc - Ft + Jt@nlp_vval)

    if opts['print_level'] > 2: print('- elapsed time:',oc.now() - initial_timestamp)




######################################################################################################
########################### -------------- Start Iterations -------------- ###########################
######################################################################################################

    # store the original lower and upper bounds for the variables in the problem
    lob = oc.DM(local_problem.var['lob'])
    upb = oc.DM(local_problem.var['upb'])

    # create a copy of the variable bounds to fix the discrete variables
    lob_fix = oc.DM(local_problem.var['lob'])
    upb_fix = oc.DM(local_problem.var['upb'])

    # create a process to run the linear bound propagation
    if opts['bound_propagation']:
        lbp_notification = Event()
        lbp_pipe1, lbp_pipe2  = Pipe()
        lbp_process = Process(target=oc.concurrent_linear_bound_propagation,args=(A,l,u,lob,upb,lbp_pipe2,lbp_notification,{'dsc':dsc_i,'tolerance':opts['primal_tol']}))
        lbp_process.start()


    # define the initial values for the variables used in the iterations
    num_iteration = 0
    elapsed_time = oc.now() - initial_timestamp
    nlp_pval = integrated_meas = 0
    alpha = 1

    lob[0] = nlp_vval[0]
    mip_vval = oc.deepcopy(nlp_vval)
    mip_vval[dsc_i] = oc.dm_round(mip_vval[dsc_i])
    last_assignment = mip_vval[dsc_i]

    incumbent_oval = oc.DM.inf()
    rel_gap = oc.DM.inf()
    abs_gap = oc.DM.inf()

    # algorithm modifiers
    disable_proximal_step = (opts['algorithm'] == -2) or not opts['with_proximal_step']

    # store the current discrete assignment to check cycling
    if opts['check_cycling']: past_ass = [mip_vval[dsc_i]]

    # start the main loop
    while (opts['max_iter'] is None or num_iteration <= opts['max_iter']) and\
          (opts['max_time'] is None or elapsed_time < opts['max_time']):

        num_iteration += 1
        elapsed_time = oc.now() - initial_timestamp

        # print progress to screen
        if opts['print_level'] > 1:
            print('\n---------------------------------------------------------------------------------')
            print('Iteration :',num_iteration,'- Time:',round(elapsed_time,2),'- Alpha:',alpha,'- Incubent Objective:',pp_remap['oval'](incumbent_oval),(abs_gap,rel_gap))
            print('---------------------------------------------------------------------------------\n')
        elif opts['print_level'] > 0:
            print('- Iteration :',num_iteration,'- Time:',round(elapsed_time,2),'- Alpha:',alpha,'- Incubent Objective:',pp_remap['oval'](incumbent_oval),(abs_gap,rel_gap))



        ################################################################################
        if opts['print_level'] > 1: print('\n+++ NLP +++')

        # variable bounds with fixed discrete variables
        upb_fix[0] = upb[0]
        lob_fix[dsc_i] = mip_vval[dsc_i]
        upb_fix[dsc_i] = mip_vval[dsc_i]

        if not disable_proximal_step:
            if opts['print_level'] > 1:
                print(' Proximal Step:')
                timestamp_ = oc.now()


            # get the fractional feasible point that is the closest to the previous discrete assignment
            out = nlp_prx.call({'x0':mip_vval,\
                                'lob':lob,\
                                'upb':upb,\
                                'dsc_ass': mip_vval[dsc_i],\
                                'lmx0':nlp_vlam,\
                                'lmg0':nlp_clam})
            nlp_vval = oc.DM(out['var'])
            nlp_vlam = oc.DM(out['lmx'])
            nlp_clam = oc.DM(out['lmg'])

            if out['var'][0] > upb[0]: # if the proximal step resulted infeasible, exit
                if opts['print_level'] > 1: print('- status: Failure')
                break

            # store the optimal solution of the nlp
            if opts['nl_solver_norm'] == 'L2':
                nlp_pval = oc.sqrt(out['obj'])
            else:
                nlp_pval = out['obj']

            # compute constraint violation
            violation_ = max(0.0,oc.amax(lob_fix-out['var']),oc.amax(out['var']-upb_fix))
            if num_lc > 0:
                Flc_ = Flc.call({'i':out['var']})['o']
                violation_ = max(violation_,oc.amax(Llc-Flc_),oc.amax(Flc_-Ulc))
            if num_nlc > 0:
                Fnlc_ = Fnlc.call({'i':out['var']})['o']
                violation_ = max(violation_,oc.amax(Lnlc-Fnlc_),oc.amax(Fnlc_-Unlc))

            # check feasibility
            if violation_ <= opts['primal_tol']:
                feasible = True
                if opts['print_level'] > 1:
                    print(' - status: Success')
                    print(' - objective value:',out['obj'])
            else:
                feasible = False
                if opts['print_level'] > 1:
                    print(' - status: Failure')

            if opts['print_level'] > 1:
                print(' - max violation:', violation_)
                print(' - time spent:',oc.now() - timestamp_)
        else:
            feasible = True

        # if the new assignment is feasible optimize the continuous variables
        if feasible:
            if opts['print_level'] > 1:
                print('Optimization Step:')
                timestamp_ = oc.now()

            # optimize the result fixing the value for the discrete varibles
            out = nlp_opt.call({'x0':nlp_vval,\
                                'lob':lob_fix,\
                                'upb':upb_fix,\
                                'lmx0':nlp_vlam,\
                                'lmg0':nlp_clam})

            # compute constraint violation
            violation_ = max(0.0,oc.amax(lob_fix-out['var']),oc.amax(out['var']-upb_fix))
            if num_lc > 0:
                Flc_ = Flc.call({'i':out['var']})['o']
                violation_ = max(violation_,oc.amax(Llc-Flc_),oc.amax(Flc_-Ulc))
            if num_nlc > 0:
                Fnlc_ = Fnlc.call({'i':out['var']})['o']
                violation_ = max(violation_,oc.amax(Lnlc-Fnlc_),oc.amax(Fnlc_-Unlc))


            # check feasibility of the solution
            if  violation_ < opts['primal_tol']:
                feasible = True
                if opts['print_level'] > 1:
                    print(' - status: Success')
                    print(' - objective value:',out['obj'])

            else:
                feasible = False
                if opts['print_level'] > 1: print(' - status: Failure')

            if opts['print_level'] > 1:
                print(' - max violation:', violation_)
                print(' - time spent:',oc.now() - timestamp_)


        if feasible or disable_proximal_step:
            # store the new nlp point
            nlp_vval = oc.DM(out['var'])
            nlp_vlam = oc.DM(out['lmx'])
            nlp_clam = oc.DM(out['lmg'])


        if feasible: # we have a new incumbent solution

            # store the objective improvement
            obj_improvement = pp_remap['oval'](incumbent_oval)-pp_remap['oval'](out['obj'])
            problem.var['val'] = pp_remap['vval'](nlp_vval)

            # (re)compute optimality gaps
            incumbent_oval = out['obj']
            abs_gap = pp_remap['oval'](incumbent_oval)-pp_remap['oval'](lob[0])
            rel_gap = 2*abs_gap/oc.fabs(pp_remap['oval'](incumbent_oval)+pp_remap['oval'](lob[0]))

            if opts['print_level'] > 1:
                print('\n=================================================================================')
                print(' Feasible solution found ')
                print(' - objective value:', out['obj'],'improvement:',obj_improvement)
                print(' - elapsed time:',round(oc.now() - initial_timestamp,2))
                print('=================================================================================\n')

            # stop if the desired optimality gap is reached
            if abs_gap <= opts['abs_opt_gap'] or rel_gap <= opts['rel_opt_gap']: break

            # force the next solution to be better than the current one
            upb[0] = out['obj'] - max(opts['abs_opt_gap'],1.01*opts['primal_tol'])

            # send the new objective value
            if opts['bound_propagation']:
                while lbp_notification.is_set(): oc.sleep(0.01) # wait for the pipe to be free
                lbp_pipe1.send(('bounds_update',[0],lob[0],upb[0]))
                lbp_notification.set()



        if opts['print_level'] > 1: print('\n+++ Generating New Linearizations +++')
        # prepare the new linearizations set
        Anew = oc.DM(0,num_vars)
        lnew = oc.DM()
        unew = oc.DM()

        # # if the assignment is not feasible refine the linear approximation of
        # # the feasible space and retry
        # else:
        #
        #     # add a valid inequality to avoid cycling
        #     Anew = oc.DM(1,num_vars)
        #     Anew[dsc_i] = (mip_vval[dsc_i]-nlp_vval[dsc_i]).T
        #     lnew = -oc.inf
        #     unew = (mip_vval[dsc_i]-nlp_vval[dsc_i]).T @ nlp_vval[dsc_i]

        if num_nlc > 0:
            # nonlinear constraints
            Ft = Fnlc.call({'i':nlp_vval})['o']
            Jt = oc.sparsify(Jnlc.call({'i':nlp_vval})['o'])

            if opts['limited_memory']:

                # get active constraints
                acti = oc.nz_indices(oc.sparsify((Ft<Lnlc+opts['primal_tol']) + (Ft>Unlc-opts['primal_tol'])))[0]

                # add the new constraints to the constraint set
                if len(acti):
                    Anew = oc.vertcat(Anew,Jt[acti,:])
                    lnew = oc.vertcat(lnew,Lnlc[acti]-Ft[acti]+Jt[acti,:]@nlp_vval)
                    unew = oc.vertcat(unew,Unlc[acti]-Ft[acti]+Jt[acti,:]@nlp_vval)

            else:
                # add the new constraints to the constraint set
                Anew = oc.vertcat(Anew,Jt)
                lnew = oc.vertcat(lnew,Lnlc-Ft+Jt@nlp_vval)
                unew = oc.vertcat(unew,Unlc-Ft+Jt@nlp_vval)
        else:
            Anew = oc.DM(0,num_vars)
            lnew = oc.DM()
            unew = oc.DM()

        # if the miqp resulted in a point that is too close to the
        # feasible set to be actually excuded with a hyperplane,
        # add a no-good cut to avoid cycling
        # !!!! somehow it is wrong
        # if Anew.numel() > 0:
        #     tmp_eval = Anew@nlp_vval
        #     if oc.dm_max(oc.vertcat(lnew - tmp_eval,tmp_eval - unew)) <= opts['primal_tol']:
        #         cut = oc.DM(1,A.shape[1])
        #         cut[dsc_i] = 2*mip_vval[dsc_i] -1
        #         Anew = oc.vertcat(Anew,cut)
        #         lnew = oc.vertcat(lnew,-oc.inf)
        #         unew = oc.vertcat(unew,oc.sum1(mip_vval[dsc_i])-1)
        #         print('dbg1')
        # else:
        #     print('dbg2')
        #     cut = oc.DM(1,A.shape[1])
        #     cut[dsc_i] = 2*mip_vval[dsc_i] -1
        #     Anew = oc.vertcat(Anew,cut)
        #     lnew = oc.vertcat(lnew,-oc.inf)
        #     unew = oc.vertcat(unew,oc.sum1(mip_vval[dsc_i])-1)


        # update the constraint set
        A = oc.vertcat(A,Anew)
        l = oc.vertcat(l,lnew)
        u = oc.vertcat(u,unew)



        ################################################################################
        if opts['print_level'] > 1: print('\n+++ MILP +++')

        #compute the objective gain
        if num_iteration == 1:
            norm_gain = 0
        else:
            g1 = nlp_vval[0] - rlxd_oval
            if opts['mi_solver_norm'] == 'L2':
                g2 = oc.sum1((rlxd_vval[dsc_i] - nlp_vval[dsc_i])**2)/2.0
            elif opts['mi_solver_norm'] == 'L1':
                g2 = oc.sum1(oc.fabs(rlxd_vval[dsc_i] - nlp_vval[dsc_i]))
            elif opts['mi_solver_norm'] == 'Linf':
                g2 = oc.dm_max(oc.fabs(rlxd_vval[dsc_i] - nlp_vval[dsc_i]))

            if g1 == 0:
                norm_gain = 0
            elif g2 == 0:
                norm_gain = 1e20
            else:
                norm_gain = g1/g2



        # update alpha
        if opts['algorithm'] == -2: # classical outer approximation
            alpha = 1.0

        elif opts['algorithm'] == -1: # classical outer approximation (with OFP)
            alpha = 1.0
            if incumbent_oval < oc.inf:
                disable_proximal_step = True
                alpha = 1.0
            else:
                alpha = opts['alpha_min']

        elif opts['algorithm'] == 0: # outer approximation (with proximal NLP)
            alpha = 1.0

        elif opts['algorithm'] == 1: # use the averaged projection norm (along the discrete axes) to update alpha
            integrated_meas = opts['alpha_mem']*integrated_meas + (1-opts['alpha_mem'])*nlp_pval
            alpha = opts['alpha_min'] + (1 - opts['alpha_min'])/(1 + integrated_meas/opts['infeasibility_scale'])

        elif opts['algorithm'] == 2: # use a norm weighted with the linear constraint set
            eval_ = Anew@mip_vval
            integrated_meas = opts['alpha_mem']*integrated_meas + (1-opts['alpha_mem'])*\
            sum([max(lnew[i]-eval_[i],eval_[i]-unew[i],0) for i in range(eval_.numel())])/Anew.shape[0]
            alpha = opts['alpha_min'] + (1 - opts['alpha_min'])/(1 + integrated_meas/opts['infeasibility_scale'])


        # clean up alpha
        if alpha >= 1 - 1e-2:
            alpha = 1
        elif alpha <= opts['alpha_min'] + 1e-2:
            alpha = opts['alpha_min']

        # solve the miqp
        if opts['print_level'] > 1:
            if opts['mi_solver_norm'] == 'L2':   print('- objective: '+str(oc.dm_round(alpha,4))+'*(f(_x,_y) + J(_x,_y)@[x,y]) + '+str(oc.dm_round((1-alpha)*norm_gain,4))+'*norm2(_y-y)')
            if opts['mi_solver_norm'] == 'L1':   print('- objective: '+str(oc.dm_round(alpha,4))+'*(f(_x,_y) + J(_x,_y)@[x,y]) + '+str(oc.dm_round((1-alpha)*norm_gain,4))+'*norm1(_y-y)')
            if opts['mi_solver_norm'] == 'Linf': print('- objective: '+str(oc.dm_round(alpha,4))+'*(f(_x,_y) + J(_x,_y)@[x,y]) + '+str(oc.dm_round((1-alpha)*norm_gain,4))+'*normInf(_y-y)')
            timestamp_ = oc.now()

        out = poa_mip( nlp_vval,\
                        lob,\
                        upb,\
                        dsc_i,\
                        (A,l,u),\
                        alpha,\
                        norm_gain,\
                        mip_opts)

        # store the result of the optimization
        mip_oval = out['cost']
        mip_vval = out['x']
        mip_vval[dsc_i] = oc.dm_round(mip_vval[dsc_i])

        if opts['print_level'] > 1:
            print('- time spent:',oc.now() - timestamp_)
            print('- objective value:',mip_oval)


        # check the success of the miqp
        if oc.DM(mip_oval).is_regular():

            if opts['print_level'] > 1: print('- status: Success')

            # if alpha == 1, update the objective lower bound
            if alpha == 1.0:

                # set a new lower bound
                lob[0] = mip_vval[0]

                # (re)compute optimality gaps
                if incumbent_oval < oc.DM.inf():
                    abs_gap = pp_remap['oval'](incumbent_oval)-pp_remap['oval'](lob[0])
                    rel_gap = 2*abs_gap/oc.fabs(pp_remap['oval'](incumbent_oval)+pp_remap['oval'](lob[0]))

                else:
                    abs_gap = oc.inf
                    rel_gap = oc.inf

                # if we reached the required level of optimality, exit
                if abs_gap <= 1.01*opts['abs_opt_gap'] or rel_gap <= 1.01*opts['rel_opt_gap']: break
        else:
            # if the miqp resulted infeasible, exit
            break
            # print the failure state
            if opts['print_level'] > 1: print('- status: Failure')


        # cycle dectection
        if oc.dm_max(last_assignment == mip_vval[dsc_i]) == 0:
            break
        else:
            last_assignment = mip_vval[dsc_i]

        # deep cycle detection (to remove at release)
        if opts['check_cycling']:
            for a in range(len(past_ass)):
                ass = past_ass[a]
                if oc.dm_max(oc.fabs(ass - mip_vval[dsc_i])) == 0:
                    tmp = A@mip_vval
                    print('objective value:',mip_vval[0])
                    print('constraints violation:',oc.dm_max(l-tmp), oc.dm_max(tmp-u))
                    print('bounds violation:',oc.dm_max(lob-mip_vval),oc.dm_max(mip_vval-upb),'(',mip_vval[0]-upb[0],')')
                    raise NameError('POA is caught in a cycle')
            past_ass.append(mip_vval[dsc_i])


        # update the constraint set for the bound propagation
        if opts['bound_propagation']:

            # wait for the pipe to be free
            while lbp_notification.is_set():
                oc.sleep(0.1)

            # get the current bounds
            lbp_pipe1.send(('bounds_plz',))
            lbp_notification.set()
            message = lbp_pipe1.recv()

            # store the new bounds
            lob = oc.DM(message[1]); lob_fix[cnt_i] = lob[cnt_i]
            upb = oc.DM(message[2]); upb_fix[cnt_i] = upb[cnt_i]

            #send the new constraints
            lbp_pipe1.send(('cns_update',Anew,lnew,unew))
            lbp_notification.set()



    # stop the linear bound propagation
    if opts['bound_propagation']:
        # wait for the pipe to be free
        while lbp_notification.is_set():
            oc.sleep(0.1)
        lbp_pipe1.send(('stop',))
        lbp_notification.set()
        lbp_process.join()



######################################################################################################

    # termination
    elapsed_time = oc.now() - initial_timestamp
    if opts['print_level'] > 0: print('\n=================================================================================')
    if not incumbent_oval == oc.inf:

        if not opts['max_iter'] is None and num_iteration > opts['max_iter']:
            if opts['print_level'] > 0: print('Max number of iterations reached, Success')
        elif not opts['max_time'] is None and elapsed_time > opts['max_time']:
            if opts['print_level'] > 0: print('Max time reached, Success')
        else:
            if opts['print_level'] > 0: print('Solution Found, Success')

        if opts['print_level'] > 0: print('- objective value:', pp_remap['oval'](incumbent_oval),'('+str(abs_gap)+','+str(rel_gap)+')')
        if opts['print_level'] > 0: print('- total time:',oc.now() - initial_timestamp)
        if opts['print_level'] > 0: print('=================================================================================\n')

        return {'return_status':'Success','num_iterations':num_iteration,
                'total_time':elapsed_time,'solution_time':elapsed_time - compilation_time}

    else:

        if not opts['max_iter'] is None and num_iteration > opts['max_iter']:
            if opts['print_level'] > 0: print('Max number of iterations reached, Failure')
        elif not opts['max_time'] is None and elapsed_time > opts['max_time']:
            if opts['print_level'] > 0: print('Max time reached, Failure')
        else:
            if opts['print_level'] > 0: print('Problem Infeasible, Failure')

        if opts['print_level'] > 0: print('- total time:',oc.now() - initial_timestamp)
        if opts['print_level'] > 0: print('=================================================================================\n')

        return {'return_status':'Failure','num_iterations':num_iteration,
                'total_time':elapsed_time,'solution_time':elapsed_time - compilation_time}



def POA_solvers(problem,options = None):

    opts = {}
    opts['print_level'] = 0
    opts['nl_solver_name'] = 'ipopt'
    opts['primal_tol'] = 1e-4
    opts['code_gen'] = True
    opts['integer_tol'] = None

    if not options is None:
         for k in options:
             if k in opts and not opts[k] is None:
                 opts[k] = options[k]




    # import the auxiliary scripts location in the path
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
    if cmd_folder not in sys.path:
        sys.path.insert(0, cmd_folder)


    from proximalOA_src.poa_problem_transformation import poa_problem_transformation
    from proximalOA_src.poa_nlp import poa_nlp


    # set solver options
    if opts['integer_tol'] is None:
        opts['integer_tol'] = opts['primal_tol']
    nlp_opts = {'solver':opts['nl_solver_name'],'primal_tol':opts['primal_tol'],'code_gen':opts['code_gen'],'print_level':max([0,opts['print_level']-3]),'integer_tol':opts['integer_tol']}



    # create a local copy of the problem to preprocess
    local_problem = problem.deepcopy()

    # transform the non-linear mixed-integer problem into a non-linear mixed-binary problem with linear objective
    poa_problem_transformation(local_problem)[0]

    # create the nlp solver
    return poa_nlp(local_problem,nlp_opts)
