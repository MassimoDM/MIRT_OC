#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:50:16 2017

@author: Massimo De Mauri
"""
import MIRT_OC as oc
import copy


class o_problem:

    def __init__(self,name = ''):
        self.name = name
        self.type = None
        self.var = {'nme':None,'sym':None,'lob':None,'upb':None,'dsc':None,'val':None,'lam':None,'lbl':None}
        self.cns = {'exp':None,'lob':None,'upb':None,'lam':None,'lbl':None}
        self.obj = {'exp':None,'val':None}
        self.sos1 = None

    def copy(self):
        # just call deepcopy (backward-compatibility)
        return copy.deepcopy(self)

    def deepcopy(self):

        newp = o_problem()
        newp.name = self.name

        newp.var['sym'] = oc.MX.sym('v',self.var['sym'].shape)
        newp.var['lob'] = oc.DM(self.var['lob'])
        newp.var['upb'] = oc.DM(self.var['upb'])
        if not self.var['val'] is None: newp.var['val'] = oc.DM(self.var['val'])
        if not self.var['nme'] is None: newp.var['nme'] = copy.deepcopy(self.var['nme'])
        if not self.var['dsc'] is None: newp.var['dsc'] = copy.deepcopy(self.var['dsc'])
        if not self.var['lam'] is None: newp.var['lam'] = oc.DM(self.var['lam'])
        if not self.var['lbl'] is None: newp.var['lbl'] = copy.deepcopy(self.var['lbl'])


        if not self.cns['exp'] is None: newp.cns['exp'] = oc.Function('cns',[self.var['sym']],[self.cns['exp']])(newp.var['sym'])
        if not self.cns['lob'] is None: newp.cns['lob'] = oc.DM(self.cns['lob'])
        if not self.cns['upb'] is None: newp.cns['upb'] = oc.DM(self.cns['upb'])
        if not self.cns['lam'] is None: newp.cns['lam'] = oc.DM(self.cns['lam'])
        if not self.cns['lbl'] is None: newp.cns['lbl'] = copy.deepcopy(self.cns['lbl'])

        if not self.obj['exp'] is None: newp.obj['exp'] = oc.Function('obj',[self.var['sym']],[self.obj['exp']])(newp.var['sym'])


        if not self.sos1 is None:
            newp.sos1 = []
            for s in range(len(self.sos1)):
                w_tmp = []
                for W in self.sos1[s]['w']:
                    if W is None:
                        w_tmp.append(None)
                    else:
                        wf_tmp = oc.Function('tmp',[self.var['sym']],[W])
                        w_tmp.append(wf_tmp(newp.var['sym']))

                newp.sos1.append({'g':copy.deepcopy(self.sos1[s]['g']),\
                                  'w':w_tmp})

        return newp


    def __str__(self):

        stat = self.classify()
        max_ord_cns = oc.dm_max(stat['ord_cns'])
        max_ord_obj = oc.dm_max(stat['ord_obj'])

        space = '       '

        out = 'Name: ' + self.name + '\n'

        if max_ord_cns <= 1:
            out += ' Linearly Constrained '
        elif max_ord_cns <= 2:
            out += ' Quadratically Constrained '
        else:
            out += ' Nonlinearly Constrained '


        if max_ord_obj <= 1:
            out += ' Linear Problem\n'
        elif max_ord_obj <= 2:
            out += ' Quadratic Problem\n'
        else:
            out += ' Nonlinear Problem\n'




        out += ' - ' + str(self.var['sym'].numel())+ ' variables:\n'
        out += space + str(stat['n_dsc'])+' discrete ('+ str(stat['n_bin']) +' binary)\n'
        out += space + str(stat['n_cnt'])+' continuous\n\n'

        out+= '- ' + str(self.cns['exp'].numel()) + ' constraints:\n'

        out += space + str(stat['n_ccns']) + ' constant\n'
        out += space + str(stat['n_lcns']) + ' linear\n'
        out += space + str(stat['n_qcns']) + ' quadratic\n'
        out += space + str(stat['n_nlcns']) + ' general non-linear\n'

        return out


    def append_var_beg(self,sym,lob,upb,val,dsc=False,lam=0,lbl=''):

        if not self.var['nme'] is None: [sym.name()] + self.var['nme']
        self.var['sym'] = oc.vertcat(sym,self.var['sym'])
        self.var['lob'] = oc.vertcat(lob,self.var['lob'])
        self.var['upb'] = oc.vertcat(upb,self.var['upb'])
        self.var['val'] = oc.vertcat(val,self.var['val'])

        if not self.var['dsc'] is None:
            self.var['dsc'] = [dsc] + self.var['dsc']
        else:
            self.var['dsc'] = [dsc] + [False]*(self.var['sym'].numel()-1)


        if not self.var['lam'] is None:
            self.var['lam'] = oc.vertcat(lam,self.var['lam'])
        else:
            self.var['lam'] = oc.vertcat(lam,oc.DM.zeros(self.var['sym'].numel()-1))

        if not self.var['lbl'] is None: self.var['lbl'] = [lbl] + self.var['lbl']


    def append_var_end(self,sym,lob,upb,val,dsc=False,lam=0,lbl=''):

        if not self.var['nme'] is None: self.var['nme'].append(sym.name())
        self.var['sym'] = oc.vertcat(self.var['sym'],sym)
        self.var['lob'] = oc.vertcat(self.var['lob'],lob)
        self.var['upb'] = oc.vertcat(self.var['upb'],upb)
        self.var['val'] = oc.vertcat(self.var['val'],val)

        if not self.var['dsc'] is None:
            self.var['dsc'].append(dsc)
        else:
            self.var['dsc'] = [False]*(self.var['sym'].numel()-1) + [dsc]


        if not self.var['lam'] is None:
            self.var['lam'] = oc.vertcat(self.var['lam'],lam)
        else:
            self.var['lam'] = oc.vertcat(oc.DM.zeros(self.var['sym'].numel()-1),lam)

        if not self.var['lbl'] is None: self.var['lbl'].append(lbl)




    def append_cns_beg(self,exp,lob,upb,lam=0,lbl=''):
        self.cns['exp'] = oc.vertcat(exp,self.cns['exp'])
        self.cns['lob'] = oc.vertcat(lob,self.cns['lob'])
        self.cns['upb'] = oc.vertcat(upb,self.cns['upb'])
        if not self.cns['lam'] is None:
            self.cns['lam'] = oc.vertcat(lam,self.cns['lam'])
        else:
            self.cns['lam'] = oc.vertcat(lam,oc.DM.zeros(self.cns['exp'].numel()-1))

        if not self.cns['lbl'] is None : self.cns['lbl'] = [lbl]+self.cns['lbl']


    def append_cns_end(self,exp,lob,upb,lam=0,lbl=''):
        self.cns['exp'] = oc.vertcat(self.cns['exp'],exp)
        self.cns['lob'] = oc.vertcat(self.cns['lob'],lob)
        self.cns['upb'] = oc.vertcat(self.cns['upb'],upb)
        if not self.cns['lam'] is None:
            self.cns['lam'] = oc.vertcat(self.cns['lam'],lam)
        else:
            self.cns['lam'] = oc.vertcat(oc.DM.zeros(self.cns['exp'].numel()-1),lam)

        if not self.cns['lbl'] is None : self.cns['lbl'] = self.cns['lbl'] + [lbl]







    def get_grouped_variables_value(self,nameset = None):

        if nameset is None: nameset = list(set(self.var['nme']))

        out = {}
        for n in nameset:
            out[n] = self.var['val'][[i for i in range(len(self.var['nme'])) if self.var['nme'][i] == n]]

        return(out)


    def get_constraints_value(self):
        return oc.Function('F',[self.var['sym']],[self.cns['exp']])(self.var['val'])

    def get_objective_value(self):
        return oc.Function('F',[self.var['sym']],[self.obj['exp']])(self.var['val'])


    def get_violated_cns(self,tol):
        return oc.nz_indices(oc.Function('F',[self.var['sym']],[((self.cns['exp']<self.cns['lob']-tol)+(self.cns['exp']>self.cns['upb']+tol))>0])(self.var['val']))[0]

    def get_active_cns(self):
        return oc.find_nz(oc.Function('F',[self.var['sym']],[((self.cns['exp']<=self.cns['lob'])+(self.cns['exp']>=self.cns['upb']))>0])(self.var['val']))[0]

    def get_out_of_bound_vars(self,tol):
        return oc.find_nz((self.var['val']<self.var['lob']-tol) + (self.var['val']>self.var['upb']+tol)>0)

    def get_active_bounds(self):
        return oc.find_nz((self.var['val']<=self.var['lob']) + (self.var['val']>=self.var['upb'])>0)[0]

    def get_max_violation(self):
        out = {}
        if self.cns['exp'] is None or self.cns['exp'].numel() == 0:
            out['bounds'] =  oc.dm_max(oc.vertcat(self.var['lob']-self.var['val'],self.var['val']-self.var['upb'],0))
            out['constraints'] = 0
        else:
            out['bounds'] = oc.dm_max(oc.vertcat(self.var['lob']-self.var['val'],self.var['val']-self.var['upb'],0))
            cnsval = oc.Function('F',[self.var['sym']],[self.cns['exp']])(self.var['val'])
            out['constraints'] = oc.dm_max(oc.vertcat(self.cns['lob']-cnsval,cnsval-self.cns['upb'],0))
        return out

    def print_status(self):
        violation = self.get_max_violation()
        print('Max bounds violation: ' + str(violation['bounds']))
        print('Max constraints violation: ' + str(violation['constraints']))
        print('Objective value: ' + str(self.get_objective_value()))





    def classify(self):
        out = {}
        # get order of constraints and objective
        orders = self.get_orders()
        out['ord_obj'] = orders['obj']
        out['ord_cns'] = orders['cns']

        # classify the variables
        out['n_dsc'] = sum(self.var['dsc'])
        out['n_bin'] = len([i for i in range(len(self.var['dsc'])) if self.var['dsc'][i] == 1 and self.var['lob'][i] ==0 and self.var['upb'][i] == 1])
        out['n_cnt'] = self.var['sym'].numel() - out['n_dsc']

        # classify the constraints
        out['n_ccns']  = len([i for i in range(out['ord_cns'].numel()) if  out['ord_cns'][i]==0])
        out['n_lcns']  = len([i for i in range(out['ord_cns'].numel()) if  out['ord_cns'][i]==1])
        out['n_qcns']  = len([i for i in range(out['ord_cns'].numel()) if  out['ord_cns'][i]==2])
        out['n_nlcns'] = len([i for i in range(out['ord_cns'].numel()) if  out['ord_cns'][i]==3])


        return out




    def get_dependency_maps(self,which='all'):

        if which == 'all' or which == 'cns':
            dmap_cns = oc.sparsify(oc.DM())
            for i in range(self.var['sym'].numel()):
                dmap_cns = oc.horzcat(dmap_cns,oc.sparsify(oc.which_depends(self.cns['exp'],self.var['sym'][i],1,True)))

        if which == 'all' or which == 'obj':
            dmap_obj = oc.sparsify(oc.DM())
            for i in range(self.var['sym'].numel()):
                dmap_obj = oc.horzcat(dmap_obj,oc.sparsify(oc.which_depends(self.obj['exp'],self.var['sym'][i],1,True)))

        if which == 'all':
            return {'cns':dmap_cns,'obj':dmap_obj}
        elif which == 'cns':
            return dmap_cns
        elif which == 'obj':
            return dmap_obj
        else:
            raise NameError('Option unknown')


    def get_constant_cns(self):

        (x,f) = oc.MX2SX(self.var['sym'],self.cns['exp'])
        dep = oc.which_depends(f,x,1,True)
        return [k for k in range(len(dep)) if dep[k] == False]


    def get_linear_cns(self):

        (x,f) = oc.MX2SX(self.var['sym'],self.cns['exp'])
        dep1 = oc.which_depends(f,x,1,True)
        dep2 = oc.which_depends(f,x,2,True)
        return [k for k in range(self.cns['exp'].numel()) if dep1[k] == True and dep2[k] == False]




    def get_quadratic_cns(self):

        # get non linear constraints
        nl = self.get_non_linear_cns()


        # quadratic constraints have linear jacobian entries
        if len(nl):
            J = oc.jacobian(self.cns['exp'][nl],self.var['sym'])
            (x,f) = oc.MX2SX(self.var['sym'],J)
            dep = oc.which_depends(f,x,2,True)
            dep = oc.nz_indices(oc.sparsify(oc.sum2(oc.DM(J.sparsity(),dep))<1))[0]

            return [nl[k] for k in dep]
        else:
            return []


    def get_non_linear_cns(self):

        (x,f) = oc.MX2SX(self.var['sym'],self.cns['exp'])
        dep = oc.which_depends(f,x,2,True)

        return [k for k in range(len(dep)) if dep[k] == True]



    def get_orders(self,which='all'):

        if which == 'all' or which == 'cns':

            order_cns = oc.DM.zeros(self.cns['exp'].numel())

            # get non constant constraints
            (x,f) = oc.MX2SX(self.var['sym'],self.cns['exp'])
            dep1 = oc.which_depends(f,x,1,True)
            dep1 = [k for k in range(self.cns['exp'].numel()) if dep1[k]]
            order_cns[dep1] = 1

            # get non-linear constraints
            dep3 = oc.which_depends(f[dep1],x,2,True)
            dep3 = [dep1[k] for k in range(len(dep1)) if dep3[k]]
            order_cns[dep3] = 3

            # get quadratic constraints
            if len(dep3):
                J = oc.jacobian(self.cns['exp'][dep3],self.var['sym'])
                (x,f) = oc.MX2SX(self.var['sym'],J)
                dep2 = oc.which_depends(f,x,2,True)
                dep2 = oc.nz_indices(oc.sparsify(oc.sum2(oc.DM(J.sparsity(),dep2))<1))[0]
                dep2 = [dep3[k] for k in dep2]
                order_cns[dep2] = 2

        if which == 'all' or which == 'obj':

            order_obj = oc.DM.zeros(self.obj['exp'].numel())

            # get non constant constraints
            (x,f) = oc.MX2SX(self.var['sym'],self.obj['exp'])
            dep1 = oc.which_depends(f,x,1,True)
            dep1 = [k for k in range(self.obj['exp'].numel()) if dep1[k]]
            order_obj[dep1] = 1

            # get non-linear constraints
            dep3 = oc.which_depends(f[dep1],x,2,True)
            dep3 = [dep1[k] for k in range(len(dep1)) if dep3[k]]
            order_obj[dep3] = 3

            # get quadratic constraints
            if len(dep3):
                J = oc.jacobian(self.obj['exp'][dep3],self.var['sym'])
                (x,f) = oc.MX2SX(self.var['sym'],J)
                dep2 = oc.which_depends(f,x,2,True)
                dep2 = oc.nz_indices(oc.sparsify(oc.sum2(oc.DM(J.sparsity(),dep2))<1))[0]
                dep2 = [dep3[k] for k in dep2]
                order_obj[dep2] = 2



        if which == 'all':
            return {'cns':order_cns,'obj':order_obj}
        elif which == 'cns':
            return order_cns
        elif which == 'obj':
            return order_obj
        else:
            raise NameError('Option unknown')



    def get_dsc_vars(self):
        return [i for i in range(len(self.var['dsc'])) if self.var['dsc'][i]]

    def get_cnt_vars(self):
        return [i for i in range(len(self.var['dsc'])) if not self.var['dsc'][i]]

    def get_bin_vars(self):
        return [i for i in range(len(self.var['dsc'])) if self.var['dsc'][i] and self.var['lob'][i] == 0 and self.var['upb'][i] == 1]

    def linearize(self, lin_point=None):


        nlc_i = self.get_non_linear_cns()

        if lin_point is None:
            lin_point = self.var['val']

        obj_f = oc.Function('obj',[self.var['sym']],[self.obj['exp']])
        obj_J = oc.Function('Jobj',[self.var['sym']],[oc.jacobian(self.obj['exp'],self.var['sym'])])
        self.obj['exp'] = obj_f(lin_point) + obj_J(lin_point)@(self.var['sym']-lin_point)

        cns_f = oc.Function('obj',[self.var['sym']],[self.cns['exp'][nlc_i]])
        cns_J = oc.Function('Jobj',[self.var['sym']],[oc.jacobian(self.cns['exp'][nlc_i],self.var['sym'])])
        self.cns['exp'][nlc_i] = cns_f(lin_point) + cns_J(lin_point)@(self.var['sym']-lin_point)

    def get_linearized_obj(self,linpoint=None):

        if linpoint is None:
            linpoint = self.var['val']


        f = oc.substitute(self.obj['exp'],self.var['sym'],linpoint)
        J = oc.substitute(oc.jacobian(self.obj['exp'],self.var['sym']),self.var['sym'],linpoint)

        return f+J@(self.var['sym']-linpoint)



    def get_linearized_cns(self, cns_list = None, lin_point=None):



        if cns_list is None:
            cns_list = list(range(self.cns['exp'].numel()))

        if lin_point is None:
            lin_point = self.var['val']

        tmp_cns = {}
        tmp_cns['exp'] = self.cns['exp'][cns_list]
        tmp_cns['lob'] = self.cns['lob'][cns_list]
        tmp_cns['upb'] = self.cns['upb'][cns_list]

        if not self.cns['lam'] is None:
            tmp_cns['lam'] = self.cns['lam'][cns_list]
        else:
            tmp_cns['lam'] = oc.DM.zeros(len(cns_list))

        if not self.cns['lbl'] is None: tmp_cns['lbl'] = [self.cns['lbl'][i] for i in cns_list]

        nlc_i_full = self.get_non_linear_cns()
        nlc_i = [i for i in range(len(cns_list)) if cns_list[i] in nlc_i_full]

        if len(nlc_i):
            cns_f = oc.Function('cns',[self.var['sym']],[tmp_cns['exp'][nlc_i]])
            cns_J = oc.Function('Jcns',[self.var['sym']],[oc.jacobian(tmp_cns['exp'][nlc_i],self.var['sym'])])

        tmp_cns['exp'][nlc_i] = cns_f(lin_point) + cns_J(lin_point)@(self.var['sym']-lin_point)


        return tmp_cns


    def get_sos1(self):

        # sos1 constraints are linear
        cns_list = self.get_linear_cns()

        # sos1 constraints are equality constraints
        cns_list = [i for i in cns_list if self.cns['lob'][i] == self.cns['upb'][i]]

        # sos1 constraints depends only on binary variables
        bin_i = self.get_bin_vars()
        non_bin_i = [i for i in range(self.var['sym'].numel()) if not i in bin_i]


        tmp =  oc.which_depends(self.cns['exp'][cns_list],self.var['sym'][non_bin_i],1,True)
        cns_list = [cns_list[i] for i in range(len(cns_list)) if tmp[i] is False]

        # sos1 constraints evaluated in zero give -1
        tmp_f = oc.Function('f',[self.var['sym'][bin_i]],[self.cns['exp'][cns_list]-self.cns['lob'][cns_list]])
        tmp = tmp_f(oc.zeros(self.var['sym'][bin_i].numel()))
        cns_list = [cns_list[i] for i in range(len(cns_list)) if tmp[i] == -1]


        # the jacobian of sos1 constraints is [1, 1, 1, ...
        tmp_f = oc.Function('j',[self.var['sym'][bin_i]],[oc.jacobian(self.cns['exp'][cns_list],self.var['sym'][bin_i])])
        tmp = tmp_f(oc.zeros(self.var['sym'][bin_i].numel()))
        sp = tmp != oc.zeros(tmp.shape)
        sp = [[j for j in range(sp.shape[1]) if sp[i,j] != 0 ] for i in range(sp.shape[0])]
        good = [i for i in range(len(cns_list)) if oc.amax(tmp[i,sp[i]]) == 1 and oc.amin(tmp[i,sp[i]]) == 1]

        sp = [sp[i] for i in good]

        groups = [[bin_i[sp[i][j]] for j in range(len(sp[i]))] for i in range(len(sp))]

        return groups

    def get_wsos1(self):

         # sos1 constraints are linear
        cns_list = self.get_linear_cns()

        # sos1 constraints are equality constraints
        cns_list = [i for i in cns_list if self.cns['lob'][i] == self.cns['upb'][i]]

        # sos1 constraints depends only on binary variables
        bin_i = self.get_bin_vars()
        non_bin_i = [i for i in range(self.var['sym'].numel()) if not i in bin_i]


        tmp =  oc.which_depends(self.cns['exp'][cns_list],self.var['sym'][non_bin_i],1,True)
        cns_list = [cns_list[i] for i in range(len(cns_list)) if tmp[i] is False]

        # sos1 constraints evaluated in zero give -1
        tmp_f = oc.Function('f',[self.var['sym'][bin_i]],[self.cns['exp'][cns_list]-self.cns['lob'][cns_list]])
        tmp = tmp_f(oc.zeros(self.var['sym'][bin_i].numel()))
        cns_list = [cns_list[i] for i in range(len(cns_list)) if tmp[i] == -1]


        # the jacobian of sos1 constraints is [1, 1, 1, ...
        tmp_f = oc.Function('j',[self.var['sym'][bin_i]],[oc.jacobian(self.cns['exp'][cns_list],self.var['sym'][bin_i])])
        tmp = tmp_f(oc.zeros(self.var['sym'][bin_i].numel()))
        sp = tmp != oc.zeros(tmp.shape)
        sp = [[j for j in range(sp.shape[1]) if sp[i,j] != 0 ] for i in range(sp.shape[0])]
        good = [i for i in range(len(cns_list)) if oc.amax(tmp[i,sp[i]]) == 1 and oc.amin(tmp[i,sp[i]]) == 1]

        sp = [sp[i] for i in good]

        groups = [[bin_i[sp[i][j]] for j in range(len(sp[i]))] for i in range(len(sp))]


    def linear_bound_propagation(self,options=None):

        if not self.var['dsc'] is None:
            if options is None:
                options = {}
            if not 'dsc' in options:
                options['dsc'] = self.get_dsc_vars()

        lin_i = self.get_linear_cns()


        cns = self.cns['exp'][lin_i]
        F = oc.Function('j',[self.var['sym']],[cns])(oc.DM.zeros(self.var['sym'].numel()))
        J = oc.Function('j',[self.var['sym']],[oc.jacobian(cns,self.var['sym'])])(oc.DM.zeros(self.var['sym'].numel()))







        self.var['lob'], self.var['upb'] = oc.linear_bound_propagation(
                                                        J,\
                                                       self.cns['lob'][lin_i] - F,\
                                                       self.cns['upb'][lin_i] - F,\
                                                       self.var['lob'],\
                                                       self.var['upb'],\
                                                       options)
        return


    def eliminate_fixed_vars(self,tolerance=0,verbose = False):

        nvar = self.var['sym'].numel()

        to_keep = []
        to_subs = []
        values = []


        for k in range(nvar):
            if self.var['upb'][k] - self.var['lob'][k] <= tolerance:
                to_subs.append(k)
                values.append(.5*(self.var['upb'][k] + self.var['upb'][k]))
            else:
                to_keep.append(k)


        new_vars = oc.MX.sym('V',len(to_keep))
        tmp = oc.MX(nvar,1)
        tmp[to_keep] = new_vars
        tmp[to_subs] = oc.DM(values)

        self.cns['exp'] = oc.substitute(self.cns['exp'],self.var['sym'],tmp)
        self.obj['exp'] = oc.substitute(self.obj['exp'],self.var['sym'],tmp)

        self.var['sym'] = new_vars
        self.var['lob'] = self.var['lob'][to_keep]
        self.var['upb'] = self.var['upb'][to_keep]


        if not self.var['nme'] is None: self.var['nme'] = [self.var['nme'][k] for k in to_keep]
        if not self.var['dsc'] is None: self.var['dsc'] = [self.var['dsc'][k] for k in to_keep]
        if not self.var['lbl'] is None: self.var['lbl'] = [self.var['lbl'][k] for k in to_keep]

        if not self.var['val'] is None: self.var['val'] = self.var['val'][to_keep]
        if not self.var['lam'] is None: self.var['lam'] = self.var['lam'][to_keep]

        return


    def append(self,problem):


        if problem.var['sym'] is None:
            return

        if self.var['sym'] is None:

            self.var  = problem.var
            self.cns  = problem.cns
            self.sos1 = problem.sos1
            self.obj  = problem.obj

        else:

            import re
            numV1 = self.var['sym'].numel()
            numV2 = problem.var['sym'].numel()




            # collect parameters, dynamic states and discrete transition states
            PXZ_nme = []

            flag = False
            PXZ_i1 = []
            for k in reversed(range(numV1)):

                if '<ocp_p>' in self.var['lbl'][k] or\
                   '<ocp_x>' in self.var['lbl'][k] or\
                   '<ocp_z>' in self.var['lbl'][k] :

                    flag = True
                    PXZ_nme.append(self.var['nme'][k])
                    PXZ_i1.append(k)

                elif flag:
                    break

            flag = False
            PXZ_i2 = [-1]*len(PXZ_i1)
            for k in range(numV2):
                if '<ocp_p>' in problem.var['lbl'][k] or\
                   '<ocp_x>' in problem.var['lbl'][k] or\
                   '<ocp_z>' in problem.var['lbl'][k] :
                    flag = True
                    if problem.var['nme'][k] in PXZ_nme:
                        PXZ_i2[PXZ_nme.index(problem.var['nme'][k])] = k
                    else:
                        PXZ_nme.append(problem.var['nme'][k])
                        PXZ_i1.append(-1)
                        PXZ_i2.append(k)


                elif flag:
                    break

            # keep the common variables only
            k = 0
            while k < len(PXZ_nme):
                if PXZ_i1[k] == -1 or PXZ_i2[k] == -1:
                    del(PXZ_nme[k])
                    del(PXZ_i1[k])
                    del(PXZ_i2[k])
                else:
                    k += 1

            numVc = len(PXZ_nme)
            PXZ_i2_compl = [k for k in range(numV2) if not k in PXZ_i2]


            # create a new array of variablesol
            new_sym = oc.MX.sym('v',numV1 + numV2 - numVc)

            # map the new variables on the old variables they replace
            old_sym = oc.vertcat(self.var['sym'],problem.var['sym'])

            sym_map = oc.MX(old_sym.shape)
            sym_map[:numV1] = new_sym[:numV1]
            sym_map[[numV1+k for k in PXZ_i2_compl]] = new_sym[numV1:]
            sym_map[[numV1+k for k in PXZ_i2]] = sym_map[PXZ_i1]

            # collect data for the new variables
            new_lob = oc.DM(new_sym.shape)
            new_upb = oc.DM(new_sym.shape)
            new_val = oc.DM(new_sym.shape)
            new_lam = oc.DM(new_sym.shape) # to do in a later moment
            new_nme = ['']*(numV1 + numV2 - numVc)
            new_lbl = ['']*(numV1 + numV2 - numVc)
            new_dsc = [False]*(numV1 + numV2 - numVc)


            # update variables data
            self.var['sym'] = new_sym
            self.var['lob'] = new_lob
            self.var['upb'] = new_upb
            self.var['val'] = new_val
            self.var['lam'] = new_lam
            self.var['nme'] = new_nme
            self.var['lbl'] = new_lbl
            self.var['dsc'] = new_dsc




            for k in range(new_sym.numel()):
                if k < numV1:
                    if k in PXZ_i1:

                        i = PXZ_i1.index(k)

                        new_nme[k] = self.var['nme'][PXZ_i1[i]]
                        new_lob[k] = max([self.var['lob'][PXZ_i1[i]],problem.var['lob'][PXZ_i2[i]]])
                        new_upb[k] = min([self.var['upb'][PXZ_i1[i]],problem.var['upb'][PXZ_i2[i]]])
                        new_val[k] = min([new_upb[k],max([new_lob[k],.5*(self.var['val'][PXZ_i1[i]]+problem.var['val'][PXZ_i2[i]])])])
                        new_lam[k] = 0
                        new_dsc[k] = self.var['dsc'][PXZ_i1[i]] or problem.var['dsc'][PXZ_i2[i]]
                        new_lbl[k] = ''.join(set(re.findall('<.+?>',self.var['lbl'][PXZ_i1[k]])).union(re.findall('<.+?>',problem.var['lbl'][PXZ_i2[k]])))

                    else:

                        new_nme[k] = self.var['nme'][k]
                        new_lob[k] = self.var['lob'][k]
                        new_upb[k] = self.var['upb'][k]
                        new_val[k] = self.var['val'][k]
                        new_lam[k] = 0
                        new_dsc[k] = self.var['dsc'][k]
                        new_lbl[k] = self.var['lbl'][k]

                else:

                    new_nme[k] = problem.var['nme'][PXZ_i2_compl[k-numV1]]
                    new_lob[k] = problem.var['lob'][PXZ_i2_compl[k-numV1]]
                    new_upb[k] = problem.var['upb'][PXZ_i2_compl[k-numV1]]
                    new_val[k] = problem.var['val'][PXZ_i2_compl[k-numV1]]
                    new_lam[k] = 0
                    new_dsc[k] = problem.var['dsc'][PXZ_i2_compl[k-numV1]]
                    new_lbl[k] = problem.var['lbl'][PXZ_i2_compl[k-numV1]]





            # concatenate the vector of constraints
            if self.cns['exp'] is None:
                self.cns = problem.cns

            elif not problem.cns['exp'] is None:


                self.cns['exp'] = oc.substitute(oc.vertcat(self.cns['exp'],problem.cns['exp']),old_sym,new_sym)
                self.cns['lob'] = oc.vertcat(self.cns['lob'],problem.cns['lob'])
                self.cns['upb'] = oc.vertcat(self.cns['upb'],problem.cns['upb'])


                # ensure coherence of the lagrangian multipliers
                if not self.cns['lam'] is None and not problem.cns['lam'] is None:
                    self.cns['lam'] = oc.vertcat(self.cns['lam'],problem.cns['lam'])

                elif self.cns['lam'] is None and not problem.cns['lam'] is None:
                    self.cns['lam'] = oc.vertcat(oc.zeros(self.cns['exp'].numel()),problem.cns['lam'])

                elif not self.cns['lam'] is None and problem.cns['lam'] is None:
                    problem.cns['lam'] = oc.vertcat(self.cns['lam'],oc.zeros(problem.cns['exp'].numel()))

                else:
                    problem.cns['lam'] = None


                # ensure coherence of the labels
                if not self.cns['lbl'] is None and not problem.cns['lbl'] is None:
                    self.cns['lbl'] = self.cns['lbl'] + problem.cns['lbl']

                elif self.cns['lbl'] is None and not problem.cns['lbl'] is None:
                    self.cns['lbl'] = ['']*self.cns['exp'].numel() + problem.cns['lbl']

                elif not self.cns['lbl'] is None and problem.cns['lbl'] is None:
                    problem.cns['lbl'] = self.cns['lbl'] + ['']*problem.cns['exp'].numel()

                else:
                    problem.cns['lbl'] = None





            # concatenate the objective
            self.obj['exp'] = oc.vertcat(oc.substitute(self.obj['exp'],old_sym,new_sym),oc.substitute(problem.obj['exp'],old_sym,new_sym))
