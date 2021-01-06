# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-04T14:15:40+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: bonmin_interface.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-04T18:13:28+01:00
# @License: LGPL-3.0

import casadi as cs
from warnings import warn
from re import search as search_in_string


class Bonmin_workspace:

    def __init__(self,problem,options=None,additional_parameters=None):
        self.problem = problem

        # default options
        self.options = {}
        self.options['algorithm'] = 'OA'
        self.options['linear_solver'] = 'ma27'
        self.options['mi_solver_name'] = 'Cbc'
        self.options['nl_solver_name'] = 'Ipopt'
        self.options['printLevel'] = 0
        self.options['max_time'] = None
        self.options['max_iter'] = None
        self.options['additional_opts'] = None
        self.options['relativeGap'] = None
        self.options['absoluteGap'] = None
        self.options['primalTolerance'] = 1e-4
        self.options['dualTolerance'] = 1e-6

        # user defined options
        if not options is None:
            for k in options:
                if k in self.options:
                    self.options[k] = options[k]
                else:
                    warn('solve_with_bonmin - option not recognized: ' + k)




        self.bonmin_options = {}
        self.bonmin_options['discrete'] = problem.var['dsc']
        self.bonmin_options['bonmin.print_level'] = min(self.options['printLevel'],1)
        self.bonmin_options['bonmin.linear_solver'] = 'ma86'
        if self.options['mi_solver_name'] in ['cplex','Cplex'] :
            self.bonmin_options['bonmin.milp_solver'] = self.options['mi_solver_name']
        elif self.options['mi_solver_name'] in ['cbc','Cbc','Cbc_D']:
            self.bonmin_options['bonmin.milp_solver'] = 'Cbc_D'
        elif self.options['mi_solver_name'] in ['cbc_par','Cbc_Par']:
            self.bonmin_options['bonmin.milp_solver'] = 'Cbc_Par'
        else:
            raise NameError('MILP subsolver unavailable')

        self.bonmin_options['bonmin.nlp_solver'] = self.options['nl_solver_name']
        self.bonmin_options['bonmin.algorithm'] = 'B-'+self.options['algorithm']
        self.bonmin_options['bonmin.cutoff_decr'] = self.options['primalTolerance']
        self.bonmin_options['bonmin.integer_tolerance'] = self.options['primalTolerance']
        self.bonmin_options['bonmin.constr_viol_tol'] = self.options['primalTolerance']
        self.bonmin_options['bonmin.dual_inf_tol'] = self.options['dualTolerance']
        if not self.options['max_iter'] is None:
            self.bonmin_options['bonmin.max_iter'] = self.options['max_iter']
        # self.bonmin_options['bonmin.milp_sub.tolerances_integrality'] = self.options['primalTolerance']
        # self.bonmin_options['bonmin.milp_sub.simplex.tolerance.feasibility'] = self.options['primalTolerance']

        # code-generate the problem solution
        # self.bonmin_options.update({'jit':True,'compiler':'shell', 'jit_options':{'flags':'-O0','compiler':'ccache gcc'}})

        # define the problem
        x0 = cs.MX.sym('x0',problem.var['sym'].numel())
        if not additional_parameters is None:
            self.with_additional_parameters = True
            prob_ = {'p':cs.vertcat(x0,additional_parameters),
                     'x':problem.var['sym'],
                     'f':problem.obj['exp'],
                     'g':problem.cns['exp']}

            arg_ = {'p':      cs.vertcat(x0,additional_parameters),
                    'x0':     x0,
                    'lbx':    cs.MX(problem.var['lob']),
                    'ubx':    cs.MX(problem.var['upb']),
                    'lbg':    cs.MX(problem.cns['lob']),\
                    'ubg':    cs.MX(problem.cns['upb'])}

            # create a solver function
            sol_ = cs.nlpsol('solver_name','bonmin', prob_,self.bonmin_options)
            out = sol_.call(arg_)

            # store the solver function
            self.solver_f = cs.Function('bonmin_solver',[x0,additional_parameters],[out['f'],out['x']],
                                        ['x0','addpar'],['f','x'])

        else:
            self.with_additional_parameters = False
            prob_ = {'p':x0,
                     'x':problem.var['sym'],
                     'f':problem.obj['exp'],
                     'g':problem.cns['exp']}

            arg_ = {'p':      x0,
                    'x0':     x0,
                    'lbx':    cs.MX(problem.var['lob']),
                    'ubx':    cs.MX(problem.var['upb']),
                    'lbg':    cs.MX(problem.cns['lob']),\
                    'ubg':    cs.MX(problem.cns['upb'])}


            # create a solver function
            sol_ = cs.nlpsol('solver_name','bonmin', prob_,self.bonmin_options)
            out = sol_.call(arg_)

            # store the solver function
            self.solver_f = cs.Function('bonmin_solver',[x0],[out['f'],out['x']],['x0'],['f','x'])



    def call(self,x0,additional_parameter_values=None):

        if self.with_additional_parameters:
            # check the input
            if additional_parameter_values is None:
                raise NameError('bonmin interface: missing value for defined additional paramenters')
                return self.solver_f.call({'x0':x0,'addpar':additional_parameter_values})
            else:
                return self.solver_f.call({'x0':x0})
