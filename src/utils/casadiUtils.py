#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 11:57:31 2017

@author: Massimo De Mauri
"""
import casadi as cs


def MX2SX(inp,out):
    var = cs.symvar(cs.vertcat(cs.reshape(inp,-1,1),cs.reshape(out,-1,1)))
    fx = cs.Function('helper', var,[inp]).expand()
    fexpr = cs.Function('helper', var,[out]).expand()
    sx_syms = fx.sx_in()
    return (fx(*sx_syms),fexpr(*sx_syms))


def list_horzcat(lin):

    if not len(lin):
        return cs.DM()

    tmp = lin[0]

    for i in lin[1:]:
        tmp = cs.horzcat(tmp,i)

    return tmp

def depends(expression,var = None):
    if var is None:
        var = cs.symvar(expression)

    out  = [cs.reshape(cs.which_depends(expression,var[i],1,True),expression.shape) for i in range(len(var))]
    return out


def dependency_mat(expression,variables = None):
    if variables is None:
        variables = cs.vertcat(*cs.symvar(expression))

    J = cs.jacobian(expression,variables)
    J_f = cs.Function('Jtmp',[variables],[J]).expand()

    return cs.DM(J_f(variables).sparsity(),1)




def improved_which_depends(expr,sym,order,transpose):
    mx_syms = cs.symvar(expr+cs.sum1(sym))
    temp = cs.Function('temp',mx_syms,[expr])
    temp = temp.expand()
    sx_syms = temp.sx_in()
    sym2 = cs.evalf(cs.jacobian(sym,cs.vcat(mx_syms))) @ cs.vcat(sx_syms)

    expression = temp.call(sx_syms)[0]
    if type(expression) == type(cs.DM()):
        if not transpose:
            return cs.DM.zeros(sym2.numel())
        else:
            return cs.DM.zeros(expression.shape)

    return cs.which_depends(expression,sym2,order,transpose)


def get_order(expression, var_list = None):


    if not expression.type_name() in ['SX','MX']:
        return cs.DM.zeros(expression.shape[0],expression.shape[1])

    if expression.numel() > 1:
        expression = cs.reshape(expression,-1,1)

    expS = expression.sparsity()


    if var_list is None:
            var_list = cs.symvar(expression)


    if len(var_list) == 0:
        return cs.DM.zeros(expression.shape)

    var = cs.vertcat(*var_list)



    orders = cs.DM.zeros(expression.shape)
    orders += cs.DM(expS,improved_which_depends(expression,var,1,True))
    orders += cs.DM(expS,improved_which_depends(expression,var,2,True))


    indxs = [i for i in range(orders.numel()) if orders[i]>1]

    if len(indxs):
        J = cs.jacobian(expression[indxs],var)

        tmp = improved_which_depends(J,var,2,True)
        # J_f = cs.Function('Jtmp',[var],[J]).expand()
        # tmp = cs.which_depends(J_f(var),var,1,True)

        S = J.sparsity()
        tmp = cs.DM(S,tmp)
        orders[indxs] += cs.sum2(tmp)>0

    return cs.DM.reshape(orders,expression.shape)


def nz_indices(exp):
    return cs.sparsify(exp).sparsity().get_triplet()

def cssum(array):
    if array.numel() == 0:
        return 0
    return cs.dot(array,cs.ones(array.shape))



def fconcat(name,left,right):

    assert(type(left) == cs.Function)
    assert(type(right) == cs.Function)
    assert(left.n_in() == right.n_out())



    tmp_rin = {}
    for k in range(right.n_in()):
        tmp_rin[right.name_in(k)] = cs.MX.sym('i'+str(k),right.numel_in(k))


    tmp_rout = right.call(tmp_rin)
    tmp_lin = {}
    for k in range(right.n_out()):
        assert(left.numel_in(k) == right.numel_out(k))
        tmp_lin[left.name_in(k)] = tmp_rout[right.name_out(k)]

    tmp_lout = left.call(tmp_lin)




    return cs.Function(name,\
                       [tmp_rin[right.name_in(k)] for k in range(right.n_in())],\
                       [tmp_lout[left.name_out(k)] for k in range(left.n_out())],\
                       [right.name_in(k) for k in range(right.n_in())],\
                       [left.name_out(k) for k in range(left.n_out())])








def is_numeric(v):
    return type(v) in [type(cs.DM())]

def is_symbolic(v):
    return type(v) in [type(cs.MX()),type(cs.SX())]
