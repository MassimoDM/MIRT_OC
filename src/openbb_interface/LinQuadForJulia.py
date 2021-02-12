# @Author: Massimo De Mauri <massimo>
# @Date:   2021-01-11T21:33:58+01:00
# @Email:  massimo.demauri@protonmail.com
# @Filename: LinQuadForJulia.py
# @Last modified by:   massimo
# @Last modified time: 2021-01-12T12:54:58+01:00
# @License: LGPL-3.0

import casadi as cs
import MIRT_OC as oc
import subprocess
import os

class LinQuadForJulia:
    def __init__(self,name,syms,expression,type='auto'):

        # info
        self.name = name
        self.sizes = (syms[0].numel(),expression.numel(),sum([s.numel() for s in syms[1:]]))
        self.main_sym = syms[0]

        # use a single parameter for the expression (because julia sucks sometimes)
        self.param_sym = oc.MX.sym("P",self.sizes[2])
        self.param_names = [s.name() for s in syms[1:]]
        expression_ = cs.substitute(expression,cs.vcat(syms[1:]),self.param_sym)

        # expression evaluation
        self.eval = cs.Function('eval',[self.main_sym,self.param_sym],[expression_])

        # expression jacobian
        jacobian = cs.jacobian(expression_,self.main_sym)
        self.jcb_nnz = jacobian.nnz()
        self.jcb_sparsity = jacobian.sparsity().get_triplet()
        self.eval_jac = cs.Function('eval_jac',[self.main_sym,self.param_sym],[jacobian])

        # hessian of each of the elements of the expression
        hessian = [cs.hessian(expression_[i],self.main_sym)[0] for i in range(expression_.shape[0])]
        self.hes_nnz = [hessian[i].nnz() for i in range(expression_.shape[0])]
        self.hes_sparsity = [hessian[i].sparsity().get_triplet() for i in range(expression_.shape[0])]
        self.eval_hes = [cs.Function('eval_hes'+str(i),[self.main_sym,self.param_sym],[hessian[i]]) for i in range(expression.shape[0])]

        # check type
        self.type = 'Linear'
        for n in self.hes_nnz:
            if n > 0:
                self.type = 'Quadratic'
                break

        if self.type == 'Quadratic' and self.type not in ['auto','Auto','Quadratic','quadratic']:
            raise NameError('LinQuadForJulia: the function is declared '+type+' but it results Quadratic.')
        if self.type == 'Linear' and self.type not in ['auto','Auto','Linear','linear']:
            raise NameError('LinQuadForJulia: the function is declared '+type+' but it results Linear.')




    def pack_for_julia(self,param_values,extra_info={}):

        num_in = self.sizes[0]
        num_out = self.sizes[1]

        input = [oc.DM.zeros(num_in),oc.vertcat(*[param_values[name]  for name in self.param_names])]

        pack = {'type':self.type}

        # compute residuals
        if num_out == 1:
            pack.update({'c':float(self.eval(*input))})

            # compute linear part
            jacobian_ = self.eval_jac(*input)
            indsL = jacobian_.sparsity().get_triplet()[1]
            valsL = jacobian_.nonzeros()
            pack.update({'L':{'inds':[i+1 for i in indsL],'vals':valsL,'n':num_in}})

            # compute quadratic part
            if self.type == 'Quadratic':
                hessian_ = self.eval_hes[0](*input)
                (colsQ,rowsQ) = hessian_.sparsity().get_triplet()
                valsQ = hessian_.nonzeros()
                pack.update({'Q':{'cols':[i+1 for i in colsQ],'rows':[i+1 for i in rowsQ],'vals':valsQ,'n':num_in,'m':num_in}})

        else:
            pack.update({'c':float(self.eval(*input))})

            # compute linear part
            jacobian_ = self.eval_jac(*input)
            (colsA,rowsA) = jacobian_.sparsity().get_triplet()
            valsA = jacobian_.nonzeros()
            pack.update({'A':{'cols':[i+1 for i in colsA],'rows':[i+1 for i in rowsA],'vals':valsA,'n':num_out,'m':num_in}})

            # compute quadratic part
            if self.type == 'Quadratic':
                data = [{}]*num_out
                for k in range(num_out):
                    hessian_ = self.eval_hes[k](*input)
                    (colsH,rowsH) = hessian_.sparsity().get_triplet()
                    valsH = hessian_.nonzeros()
                    data[k] = {'cols':[i+1 for i in colsH],'rows':[i+1 for i in rowsH],'vals':valsH,'n':num_in,'m':num_in}
                pack.update({'H':data})

        # finally return the pack
        pack.update(extra_info)
        return pack
