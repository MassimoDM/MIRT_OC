# @Author: Massimo De Mauri <massimo>
# @Date:   2020-12-30T17:17:24+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: MPCcontroller.py
# @Last modified by:   massimo
# @Last modified time: 2021-02-22T20:01:10+01:00
# @License: LGPL-3.0

import casadi as cs
import numpy as np
import MIRT_OC as oc
from warnings import warn
from time import time
from copy import copy, deepcopy

class MPCcontroller:

    def __init__(self,model,options = dict()):

        # declare object fields
        self.model = model
        self.input_vals = None
        self.options = {}
        self.model_info = {}
        self.stats = {}
        self.problem_functions = None
        self.status = "new"
        self.clib_id = -1

        # default options
        self.options['printLevel'] = 0
        self.options['hbbSettings'] = {'numProcesses':1}
        self.options['integration_opts'] = None
        self.options['max_iteration_time'] = cs.inf
        self.options['relaxed_tail_length'] = 0
        self.options['prediction_horizon_length'] = 1
        self.options['variable_parameters_exp'] = {}

        # load user options
        if not options is None:
            for k in options.keys():
                if k in self.options:
                    self.options[k] = options[k]
                else:
                    warn('Option not recognized: '+ k)


        # check the model
        syms_ = [v.sym for v in model.x]+[v.sym for v in model.y]+[v.sym for v in model.a]+[v.sym for v in model.u + model.v]
        if len(model.p) > 0 : raise NameError('MPCcontroller is not suited to optimize parameters, please fix them or give them a state dependent expression using the \'varing_parameters_exp\' option')
        if len(model.icns) > 0 : raise NameError('MPCcontroller: initial constraints not supported')
        obj_ = 0.0
        if not model.lag is None: obj_ += model.lag
        if not model.ipn is None: obj_ += model.ipn
        if not model.may is None: obj_ += model.may
        if oc.get_order(obj_,syms_) > 2 :
            raise NameError("Objectives with order higher than 2 are not currently supported. Please, perform an epigraph reformulation of the objective.")

        # prepare collection of performace statistics
        self.stats['times'] = {'total':0.,'preprocessing':0.,'iterations':{'total':0.,'pre/post-processing':0.,'minlp_solver':{'total':0.,'nlp':0.,'mip':0.,'linearizations_generation':0.}}}
        self.stats['num_solves'] = {'nlp':0,'qp':0}


        ##############################################################################################
        ####################################### Preprocessing  #######################################
        ##############################################################################################

        # start timing
        start_time = time()

        # collect info
        PHL = self.options['prediction_horizon_length']
        RTL = self.options['relaxed_tail_length']
        self.model_info['num_states'] = num_states = len(model.x)+len(model.y)
        self.model_info['num_algebraic'] = num_algebraic = len(model.a)
        self.model_info['num_controls'] = num_controls = len(model.u + model.v)
        self.model_info['num_path_constraints'] = len(model.pcns)
        num_vars_per_step = num_states + num_algebraic+ num_controls

        ############## construct the parametric mpc model ###############
        mpc_time_vector = cs.MX.sym('t',PHL+RTL+1,1)
        mpc_model = copy(model)


        # impose the initial state with a set of initial constraints
        mpc_model.icns.extend([oc.eq(v.sym,0,'<initial_state>') for v in mpc_model.x+mpc_model.y])

        # we do not support non-linear terminal costs in the shift yet (use a slack to represent the terminal cost as a terminal constraint)
        if not model.may is None and oc.get_order(model.may,syms_) > 1:
             mpc_model.a.append(oc.variable('terminal_cost_slack',-cs.DM.inf(),cs.DM.inf(),0))
             mpc_model.pcns = [oc.eq(mpc_model.a[-1].sym,0.0,'<terminal_cost_reformulation>')] + mpc_model.pcns
             mpc_model.fcns.append(oc.leq(mpc_model.may,mpc_model.a[-1].sym,'<terminal_cost_reformulation>'))
             mpc_model.may = mpc_model.a[-1].sym
             self.model_info['num_algebraic'] += 1

        # assign variables to the inputs for later definition
        for k in range(len(model.i)):
             mpc_model.i[k].val = cs.MX.sym(model.i[k].nme,PHL+RTL+1,1)
        param_symbols = [mpc_time_vector]+[i.val for i in mpc_model.i]

        # construct a parametric optimization problem describing the mpc short time horizon
        self.mpc_problem = oc.Problem()
        oc.multiple_shooting(mpc_model,self.mpc_problem,mpc_time_vector,{'integration_opts':self.options['integration_opts']},True)

        # relax the tail (if required) #TODO check what happens to the sos1 groups
        for k in range(PHL*num_vars_per_step+num_states+num_algebraic,len(self.mpc_problem.var['dsc'])):
            self.mpc_problem.var['dsc'][k] = False

        # store the parametric version of the constraint set and variable bounds
        self.problem_functions = {
            'cns': oc.CvxFuncForJulia('cns',[self.mpc_problem.var['sym']]+param_symbols,self.mpc_problem.cns['exp']),
            'obj': oc.LinQuadForJulia('obj',[self.mpc_problem.var['sym']]+param_symbols,oc.sum1(self.mpc_problem.obj['exp']))
        }

        # compile the parametric function created
        self.problem_functions['cns'].compile(oc.home_dir+'/temp_jit_files')

        # store the parametric version of the terminal part of the constraint set and the objective (for shift)
        num_constraints_per_step = num_states + len(mpc_model.pcns)
        num_steps = PHL + RTL
        self.problem_terminals = {
            'cns': oc.CvxFuncForJulia('cns',[self.mpc_problem.var['sym']]+param_symbols,self.mpc_problem.cns['exp'][num_constraints_per_step*(num_steps-1):]),
            'obj': oc.LinQuadForJulia('obj',[self.mpc_problem.var['sym']]+param_symbols,cs.sum1(self.mpc_problem.obj['exp'][num_steps-1:]))
        }


        # load the OpenBB interface
        self.openbb = oc.openbb.OpenBBinterface()
        # load the python interface of the mpc addon
        self.mpc_addon = oc.mpc_addon.MPC_addon(self.openbb)
        # load some additional helper functions in OpenBB
        self.openbb.eval_string('include(\"'+oc.home_dir+'/src/openbb_interface/helper_functions.jl\")')

        # check for multiprocessing
        if 'numProcesses' in self.options['hbbSettings'] and self.options['hbbSettings']['numProcesses'] > 1:
            # prepare for multi-processing
            current_num_procs = self.openbb.eval_string('nprocs()')
            if current_num_procs < self.options['hbbSettings']['numProcesses']:
                self.openbb.eval_string('addprocs('+str(self.options['hbbSettings']['numProcesses']-current_num_procs)+')')
            self.openbb.eval_string('@sync for k in workers() @async remotecall_fetch(Main.eval,k,:(using OpenBB)) end')

        # collect timing statistics
        self.stats['times']['preprocessing'] = time() - start_time
        self.stats['times']['total'] += self.stats['times']['preprocessing']



    def iterate(self,shift_size,measured_state,new_input_vals,mode="relaxationOnly",iterations_for_lob_recomputation=0,timeout=None):

        iteration_start_time = time()

        # collect the value for the inputs
        if self.input_vals is None:
            self.input_vals = deepcopy(new_input_vals)
        else:
            for k in self.input_vals.keys():
                print(self.input_vals[k][shift_size:],new_input_vals[k])
                self.input_vals[k] = cs.vertcat(self.input_vals[k][shift_size:],new_input_vals[k])



        # generate the problem to solve in this iteration
        for k in range(self.model_info['num_states']):
            self.mpc_problem.cns['lob'][k] = measured_state[self.mpc_problem.var['nme'][k]]
            self.mpc_problem.cns['upb'][k] = measured_state[self.mpc_problem.var['nme'][k]]

        # get the dictionaries that will be used to transfer the problem functions into julia
        cns_extra_info = {'lob':oc.cs2list(self.mpc_problem.cns['lob']),
                          'upb':oc.cs2list(self.mpc_problem.cns['upb'])}
        cns_pack = self.problem_functions['cns'].pack_for_julia(self.input_vals,cns_extra_info)
        (self.clib_id,cns_dictionary) = self.openbb.jl.load_constraint_set(cns_pack,self.clib_id)

        obj_dictionary = self.problem_functions['obj'].pack_for_julia(self.input_vals)

        var_dictionary = {"vals":oc.cs2list(self.mpc_problem.var['val']),
                          "loBs":oc.cs2list(self.mpc_problem.var['lob']),
                          "upBs":oc.cs2list(self.mpc_problem.var['upb']),
                          "dscIndices":[k+1 for (k,val) in enumerate(self.mpc_problem.var['dsc']) if val]}

        hbbSettings = self.options['hbbSettings']
        hbbSettings.update({"optimalControlInfo":(self.model_info['num_states'],
                                                  self.model_info['num_algebraic'],
                                                  self.model_info['num_controls'])})


        if self.status == 'new' or mode is None:
            # create a new model
            self.openbb.setup("HBB",{"varSet":var_dictionary,"cnsSet":cns_dictionary,"objFun":obj_dictionary},hbbSettings)
            {'total':0.,'pre/post-processing':0.,'minlp_solver':{'nlp':0.,'mip':0.,'linearizations_generation':0.}}

        else:
            [measured_state[self.mpc_problem.var['nme'][k]] for k in range(self.model_info['num_states'])]

            # shift the old model
            measured_state_vec = [measured_state[self.mpc_problem.var['nme'][k]] for k in range(self.model_info['num_states'])]
            self.mpc_addon.HBB_mpc_shift_assisted(shift_size,{"varSet":var_dictionary,"cnsSet":cns_dictionary,"objFun":obj_dictionary},
                                                     [],measured_state_vec,mode,iterations_for_lob_recomputation)

        # shift last solution
        varShift = (self.model_info['num_states']+self.model_info['num_algebraic']+self.model_info['num_controls'])
        self.mpc_problem.var['val'][:-varShift] =  self.mpc_problem.var['val'][varShift:]
        self.stats['times']['iterations']['pre/post-processing'] +=  time() - iteration_start_time

        # solve problem
        solver_start_time = time()
        status0 = self.openbb.get_status()
        numExploredNodes0 = self.openbb.workspace.mipStepWS.mipSolverWS.status.numExploredNodes
        self.openbb.solve(self.mpc_problem.var['val'])
        solution = self.openbb.get_best_feasible_node()
        status1 = self.openbb.get_status()
        nlp_time = status1['nlpTime'] - status0['nlpTime']
        mip_time = status1['mipTime'] - status0['mipTime']
        self.stats['times']['iterations']['minlp_solver']['total'] +=  time() - solver_start_time
        self.stats['times']['iterations']['minlp_solver']['nlp'] += nlp_time
        self.stats['times']['iterations']['minlp_solver']['mip'] += mip_time
        self.stats['times']['iterations']['minlp_solver']['linearizations_generation'] += status1['rlxTime'] - status0['rlxTime']
        self.stats['num_solves']['nlp'] += status1['numIterations']-status0['numIterations']
        self.stats['num_solves']['qp'] += self.openbb.workspace.mipStepWS.mipSolverWS.status.numExploredNodes-numExploredNodes0

        if len(solution)>0:
            self.status = 'solved'
            self.mpc_problem.var['val'] = cs.DM(solution["primal"])
        else:
            raise NameError('MPCcontroller: no solution found in iteration')


        iteration_time = time() - iteration_start_time
        self.stats['times']['iterations']['total'] += iteration_time
        self.stats['times']['total'] += iteration_time
        print("MPC: time per iteration =",iteration_time,"(nlp =",nlp_time,"mip =",mip_time,")")

        return (solution["objUpB"],self.mpc_problem.get_grouped_variables_value())
