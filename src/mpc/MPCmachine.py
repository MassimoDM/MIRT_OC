# @Author: Massimo De Mauri <massimo>
# @Date:   2020-12-30T17:17:24+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: MPCmachine.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-06T13:04:17+01:00
# @License: LGPL-3.0

import casadi as cs
import numpy as np
import MIRT_OC as oc
from warnings import warn
from time import time
from copy import deepcopy

class MPCmachine:

    def __init__(self,model,minlp_solver,options = dict()):

        # declare object fields
        self.model = model
        self.state_hystory = None
        self.minlp_solver = minlp_solver
        self.options = {}
        self.model_info = {}
        self.stats = {}
        self.parametric_definitions = None
        self.memory_space = {}


        # default options
        self.options['printLevel'] = 0
        self.options['minlp_solver_opts'] = None
        self.options['integration_opts'] = None
        self.options['max_iteration_time'] = cs.inf
        self.options['shifting_strategy'] = None # available: rti, conservative, hespanhol
        self.options['time_invariant_terminals'] = False
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
        syms_ = [v.sym for v in model.x]+[v.sym for v in model.z]+[v.sym for v in model.s]+[v.sym for v in model.c]
        obj_ = 0.0
        if not model.lag is None: obj_ += model.lag
        if not model.dpn is None: obj_ += model.dpn
        if not model.may is None: obj_ += model.may
        if oc.get_order(obj_,syms_) >= 2:
            raise NameError("Objectives with order higher than 2 are not currently supported. Please, perform an epigraph reformulation of the objective.")


        for v in model.p:
            if v.nme not in self.options['variable_parameters_exp'].keys():
                raise NameError('MPCmachine is not suited to optimize parameters, please fix them or give them a state dependent expression using the \'varing_parameters_exp\' option')

        if len(model.icns)>0: raise NameError('MPCmachine: initial constraints not supported')

        if self.minlp_solver != 'openbb' and (not self.options['shifting_strategy'] is None):
            raise NameError('MPCmachine: '+self.minlp_solver+' does not support the selected shifting strategy. Select \"None\" istead')



        # prepare collection of performace statistics
        self.stats['times'] = {'total':0.,'preprocessing':0.,'iterations':{'total':0.,'pre/post-processing':0.,'minlp_solver':{'nlp':0.,'mip':0.,'linearizations_generation':0.}}}
        self.stats['num_solves'] = {'nlp':0,'mip':0}


        ##############################################################################################
        ####################################### Preprocessing  #######################################
        ##############################################################################################

        # start timing
        start_time = time()

        # collect info
        PHL = self.options['prediction_horizon_length']
        RTL = self.options['relaxed_tail_length']
        self.model_info['num_states'] = num_states = len(model.x)+len(model.z)
        self.model_info['num_slacks'] = num_slacks = len(model.s)
        self.model_info['num_controls'] = num_controls = len(model.c)
        self.model_info['num_path_constraints'] = len(model.pcns)
        num_vars_per_step = num_states + num_slacks+ num_controls

        ############## construct the parametric mpc model ###############
        mpc_model = deepcopy(model)
        mpc_model.t.val = cs.MX.sym('t',PHL+RTL+1,1)

        # assign variables to the inputs for later definition
        for k in range(len(model.i)):
             mpc_model.i[k].val = cs.MX.sym(model.i[k].nme,PHL+RTL+1,1)

        # use dummy inputs to vehiculate the measured state info
        mstate_sym = {}; mstate_val = {}
        for v in mpc_model.x+mpc_model.z:
            mpc_model.i.append(oc.input(v.nme,cs.MX.sym(v.nme,PHL+RTL+1,1)))
            mstate_sym[v.nme] = mpc_model.i[-1].sym
            mstate_val[v.nme] = mpc_model.i[-1].val

        # collect the variables used to define the inputs (also the dummy ones)
        param_symbols = [mpc_model.t.val]+[i.val for i in mpc_model.i]

        # transform the variable parameters in functions of the initial state and the inputs
        if not model.p is None:
            mpc_model.p = []
            for v in model.p:
                expression_ = cs.substitute(self.options['variable_parameters_exp'][v.nme],cs.vcat([vv.sym for vv in model.x+model.z]),cs.vcat([mstate_sym[vv.nme] for vv in model.x+model.z]))
                for k in range(len(mpc_model.ode)):  mpc_model.ode[k] = cs.substitute(mpc_model.ode[k],v.sym,expression_)
                for k in range(len(mpc_model.dtr)):  mpc_model.dtr[k] = cs.substitute(mpc_model.dtr[k],v.sym,expression_)
                for k in range(len(mpc_model.pcns)): mpc_model.pcns[k].exp = cs.substitute(mpc_model.pcns[k].exp,v.sym,expression_)
                for k in range(len(mpc_model.icns)): mpc_model.icns[k].exp = cs.substitute(mpc_model.icns[k].exp,v.sym,expression_)
                for k in range(len(mpc_model.fcns)): mpc_model.fcns[k].exp = cs.substitute(mpc_model.fcns[k].exp,v.sym,expression_)
                mpc_model.lag = cs.substitute(mpc_model.lag,v.sym,expression_)
                mpc_model.dpn = cs.substitute(mpc_model.dpn,v.sym,expression_)
                mpc_model.may = cs.substitute(mpc_model.may,v.sym,expression_)

        # construct a parametric optimization problem describing the mpc short time horizon
        self.mpc_problem = oc.o_problem()
        oc.multipleShooting(mpc_model,self.mpc_problem,{'integration_opts':self.options['integration_opts']},False)

        # relax the tail (if required) #TODO check what happens to the sos1 groups
        for k in range(PHL*num_vars_per_step+num_states+num_slacks,len(self.mpc_problem.var['dsc'])):
            self.mpc_problem.var['dsc'][k] = False

        # store the parametric version of the constraint set and variable bounds
        self.parametric_definitions = {
            'cexp' : cs.Function('cexp_par',[self.mpc_problem.var['sym']]+param_symbols,[self.mpc_problem.cns['exp']]),
            'oexp' : cs.Function('oexp_par',[self.mpc_problem.var['sym']]+param_symbols,[self.mpc_problem.obj['exp']]),
        }


        self.test = {
            'cns': oc.CSfuncForJulia('cns',[self.mpc_problem.var['sym']]+param_symbols,self.mpc_problem.cns['exp']),
            'obj': oc.CSfuncForJulia('obj',[self.mpc_problem.var['sym']]+param_symbols,self.mpc_problem.obj['exp'])
        }

        # compile the parametric functions created
        self.test['cns'].compile(oc.home_dir+'/temp_jit_files')
        self.test['obj'].compile(oc.home_dir+'/temp_jit_files')
        purposeful_error

        # # create the subsolver workspace
        # if self.minlp_solver == 'bonmin':
        #     self.minlp_solver_workspace = oc.Bonmin_workspace(self.mpc_problem,self.options['minlp_solver_opts'],param_symbols)
        #
        # # collect timing statistics
        # self.stats['times']['preprocessing'] = time() - start_time
        # self.stats['times']['total'] += self.stats['times']['preprocessing']



    def iterate(self,measured_state,input_vals,timeout=None):

        # collect info
        PHL = self.options['prediction_horizon_length']
        RTL = self.options['relaxed_tail_length']


        # construct the state hystory if needed
        if self.state_hystory is None:
            self.state_hystory = {}
            for v in self.model.x + self.model.z:
                self.state_hystory[v.nme] = cs.repmat(measured_state[v.nme],PHL+RTL+1,1)

        # collect parameter values
        param_vals = [input_vals['t']]+[input_vals[i.nme] for i in self.model.i]+\
                     [self.state_hystory[v.nme] for v in self.model.x+self.model.z]

        # generate the problem to solve in this iteration
        for k in range(self.model_info['num_states']):
            self.mpc_problem.var['lob'][k] = measured_state[self.mpc_problem.var['nme'][k]]
            self.mpc_problem.var['upb'][k] = measured_state[self.mpc_problem.var['nme'][k]]
        self.mpc_problem.cns['exp'] = self.parametric_definitions['cexp'](self.mpc_problem.var['sym'],*param_vals)
        self.mpc_problem.obj['exp'] = self.parametric_definitions['oexp'](self.mpc_problem.var['sym'],*param_vals)


        # solve problem
        sol_stats = oc.solve_with_bonmin(self.mpc_problem,{'print_level':1,'mi_solver_name':'cplex'})
        results = self.mpc_problem.get_grouped_variables_value()

        return results
