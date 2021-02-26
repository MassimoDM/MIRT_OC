#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:58:12 2017

@author: Massimo De Mauri
"""

import copy
import casadi as cs
import MIRT_OC as oc


class OCmodel:

    def __init__(self,name):
        self.name = name
        self.i = []                 # time varying inputs
        self.p = []                 # optimization parameters
        self.x = []                 # state
        self.y = []                 # finite state (discrte transitions)
        self.a = []                 # algebraic variables
        self.u = []                 # continuous controls
        self.v = []                 # discrete controls

        self.ode = []               # well... ODE
        self.itr = []               # discrete transitions

        self.pcns = []              # path constraints
        self.icns = []              # initial constraints
        self.fcns = []              # final constraints

        self.may = None             # end cost
        self.lag = None             # stage cost
        self.ipn = None             # disctrete stage cost

        self.t = cs.MX.sym("t")    # time vector
        self.dt = cs.MX.sym("dt")   # represents the time between one time step and the next

        self.sos1 = None
        self.wsos1 = None
        self.dsc = None
        self.ord = None



    def copy(self):


       new_model = OCmodel(self.name)
       new_model.t = self.t
       new_model.i = self.i
       new_model.p = self.p
       new_model.x = self.x
       new_model.y = self.y
       new_model.z = self.a
       new_model.u = self.u
       new_model.v = self.v

       new_model.ode = self.ode
       new_model.itr = self.itr

       new_model.pcns = self.pcns
       new_model.icns = self.icns
       new_model.fcns = self.fcns

       new_model.may = self.may
       new_model.lag = self.lag
       new_model.ipn = self.ipn

       return new_model


    def deepcopy(self):
        return copy.deepcopy(self)

    def __str__(self):

        self.classify()
        space = 3*' '
        des = space + 'Optimal Control Problem:'
        sep = '---------------------------------------------------------------------------------\n'

        out = sep +self.name + '\n'+des+'\n' + sep

        if not self.i == []:
            out += 'External inputs:\n'
            for i in self.i:
                    out += space + str(i)+'\n'
            out += '\n'

        if not self.p == []:
            out += 'Parameters:\n'
            for i in self.p:
                out += space + str(i)+'\n'
            out += '\n'

        out += 'States:\n'
        for i in self.x:
            out += space + str(i)+'\n'
        out += '\n'

        if not self.y == []:
            out += 'Instantaneous-transition states:\n'
            for i in self.y:
                out += space + str(i)+'\n'
        out += '\n'

        if not self.a == []:
            out += 'Algebraic variables:\n'
            for i in self.a:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.u == []:
            out += 'Continuous controls:\n'
            for i in self.u:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.v == []:
            out += 'Discrete controls:\n'
            for i in self.v:
                out += space + str(i)+'\n'
            out += '\n'

        out += 'Differential equation:\n'
        for i in range(len(self.ode)):
            out += space + 'dot(' + self.x[i].nme +') = '+ str(self.ode[i])+'\n'
        out +='\n'

        if not self.itr == []:
            out += 'Instantaneous transitions:\n'
            for i in range(len(self.itr)):
                out += space + self.y[i].nme + '[k+1] = ' + str(self.itr[i])+'[k] + ' +self.y[i].nme + '[k]\n'
            out += '\n'

        if not self.icns == []:
            out += 'Initial constraints:\n'
            for i in self.icns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.pcns == []:
            out += 'Path constraints:\n'
            for i in self.pcns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.fcns == []:
            out += 'Final constraints:\n'
            for i in self.fcns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.lag is None: out += 'Lagrange\'s obj.: ' + space + str(self.lag)+'\n\n'
        if not self.ipn is None: out += 'Instataneous penalty: ' + space + str(self.ipn)+'\n\n'
        if not self.may is None: out += 'Mayer\'s obj.: ' + space + str(self.may)+'\n\n'

        if not self.sos1 is None:
            out += 'SOS constraints:\n'
            for i in range(len(self.sos1)):
                out += space + str(self.sos1[i]) +'\n'


        out += '\n'

        return out



    def classify(self):

        var_list = []
        if self.p != []: var_list += [v.sym for v in self.p]
        if self.x != []: var_list += [v.sym for v in self.x]
        if self.y != []: var_list += [v.sym for v in self.y]
        if self.a != []: var_list += [v.sym for v in self.a]
        if self.u != []: var_list += [v.sym for v in self.u]
        if self.v != []: var_list += [v.sym for v in self.v]



        self.ord = 0
        if self.icns != []:
            for c in self.icns:
                c.classify(var_list)
                self.ord = max(self.ord,c.ord)

        if self.pcns != []:
            for c in self.pcns:
                c.classify(var_list)
                self.ord = max(self.ord,c.ord)

        if self.fcns != []:
            for c in self.fcns:
                c.classify(var_list)
                self.ord = max(self.ord,c.ord)



    def standardize_cns(self):

       if self.pcns != []:
           tmp = []
           for c in self.pcns:
               tmp += c.standardize()
           self.pcns = tmp

       if self.icns != []:
           tmp = []
           for c in self.icns:
               tmp += c.standardize()
           self.icns = tmp

       if self.fcns != []:
           tmp = []
           for c in self.fcns:
               tmp += c.standardize()
           self.icns = tmp


    def get_variables_value(self):

        vval_dict = {}

        if self.p != []:
            for k in range(len(self.p)):
                vval_dict[self.p[k].nme] = self.p[k].val

        if self.x != []:
            for k in range(len(self.x)):
                vval_dict[self.x[k].nme] = self.x[k].val

        if self.y != []:
            for k in range(len(self.y)):
                vval_dict[self.y[k].nme] = self.y[k].val

        if self.a != []:
            for k in range(len(self.a)):
                vval_dict[self.a[k].nme] = self.a[k].val


        if self.u != []:
            for k in range(len(self.u)):
                vval_dict[self.u[k].nme] = self.u[k].val

        if self.v != []:
            for k in range(len(self.v)):
                vval_dict[self.v[k].nme] = self.v[k].val

        return vval_dict

    # takes in a dictionary containing initial guesses for
    # the problem variables and sets their value accordingly
    def set_variables_value(self,vval_dict):

        if self.p != []:
            for k in range(len(self.p)):
                if self.p[k].nme in vval_dict:
                    self.p[k].val = vval_dict[self.p[k].nme]

        if self.x != []:
            for k in range(len(self.x)):
                if self.x[k].nme in vval_dict:
                    self.x[k].val = vval_dict[self.x[k].nme]

        if self.y != []:
            for k in range(len(self.y)):
                if self.y[k].nme in vval_dict:
                    self.y[k].val = vval_dict[self.y[k].nme]

        if self.a != []:
            for k in range(len(self.a)):
                if self.a[k].nme in vval_dict:
                    self.a[k].val = vval_dict[self.a[k].nme]


        if self.u != []:
            for k in range(len(self.u + self.v)):
                if self.u.nme in vval_dict:
                    self.u[k].val = vval_dict[self.u[k].nme]

        if self.v != []:
            for k in range(len(self.v)):
                if self.v[k].nme in vval_dict:
                    self.v[k].val = vval_dict[self.v[k].nme]

    def epigraph_reformulation(self,max_orders = [1,1,1], integration_opts = None):

        # initialize info
        reformulation_needed = False
        stage_info = oc.MX(0)
        terminal_info = oc.MX(0)

        # create list of variables to detect order with
        var_list = []
        if self.p != []: var_list += [v.sym for v in self.p]
        if self.x != []: var_list += [v.sym for v in self.x]
        if self.y != []: var_list += [v.sym for v in self.y]
        if self.a != []: var_list += [v.sym for v in self.a]
        if self.u + self.v != []: var_list += [v.sym for v in self.u + self.v]

        # reformulate each type of objective separately
        if not self.lag is None and oc.get_order(self.lag,var_list)>max_orders[0]:

            reformulation_needed = True

            # define integrator initial time
            x0 = oc.MX.sym('x0')

            # collect the symbols
            isyms = oc.vertcat(*[v.sym for v in self.i]) if self.i != [] else oc.MX()
            psyms = oc.vertcat(*[v.sym for v in self.p]) if self.p != [] else oc.MX()
            xsyms = oc.vertcat(*[v.sym for v in self.x]) if self.x != [] else oc.MX()
            ysyms = oc.vertcat(*[v.sym for v in self.y]) if self.y != [] else oc.MX()
            zsyms = oc.vertcat(*[v.sym for v in self.a]) if self.a != [] else oc.MX()
            usyms = oc.vertcat(*[v.sym for v in self.u]) if self.u != [] else oc.MX()
            vsyms = oc.vertcat(*[v.sym for v in self.v]) if self.v != [] else oc.MX()

            # evaluate integration (symbolically)
            intg = oc.integrate_ode(x0,oc.vertcat(psyms,xsyms,ysyms,zsyms,isyms,usyms,vsyms,self.t.sym),[self.lag],self.dt.sym,integration_opts)
            tmp_exp = intg.call({'x0':0,'p':oc.vertcat(psyms,xsyms,ysyms,zsyms,isyms,usyms,vsyms,self.t.sym,self.dt.sym)})['xf']

            # collect results
            stage_info += tmp_exp

            # remove the lagrangian objective
            self.lag = None

        if not self.ipn is None and oc.get_order(self.ipn,var_list)>max_orders[1]:

            reformulation_needed = True

            # collect results
            stage_info += self.ipn

            # remove the discrete penalties
            self.ipn = oc.MX(0.0)


        if not self.may is None and oc.get_order(self.may,var_list)>max_orders[2]:

            reformulation_needed = True

            # collect results
            terminal_info += self.may

            # remove the mayer term
            self.may = oc.MX(0.0)


        if reformulation_needed:

            slack_var = oc.variable('obj_slack',-oc.DM.inf(),oc.DM.inf())
            self.a.append(slack_var)

            # handle the slack in the intermediate steps
            if len(cs.symvar(stage_info)) > 0:
                self.ipn += slack_var.sym
                self.pcns.append(oc.leq(stage_info,slack_var.sym,'<epigraph_cns>'))
            else:
                self.pcns.append(oc.eq(slack_var.sym,0.0,'<epigraph_cns>'))


            # handle the slack in the terminal step
            if self.may is None: self.may = cs.MX(0.0)
            if len(cs.symvar(terminal_info)) > 0:
                self.may += slack_var.sym
                self.fcns.append(oc.leq(terminal_info,slack_var.sym,'<epigraph_cns>'))
            else:
                self.fcns.append(oc.geq(slack_var.sym,0.0,'<epigraph_cns>'))



#     # transform a mixed integer problem into a mixed binary one
#     def binarize(self,vartypes = ['p','x','y','a','c']):
#
#         subexp = {}
#         remapfnc = {}
#
#         for ty in vartypes:
#             if getattr(self,ty) != []:
#                 torem = []
#                 toadd = []
#                 if ty == 'x' or ty == 'y':
#                     new_constraints = []
#                     newvarlist = []
#
#                 for k,var in enumerate(getattr(self,ty)):
#                     if var.dsc == True and (var.lob != 0 or var.upb != 1):
#
#                         if var.upb-var.lob == 1:
#                             torem.append(k)
#                             toadd.append(oc.variable(var.nme+'_b',0,1,var.val-var.lob,True))
#                             subexp[var.nme] = toadd[-1].sym + var.lob
#
#                         else:
#
#                             if ty == 'x': new_constraints.append(oc.eq(oc.MX(0),0))
#                             if ty == 'y': new_constraints.append(oc.eq(oc.MX(0),0))
#
#                             torem.append(k)
#                             tmp_sos1 = oc.sos1_constraint([],[])
#                             tmp_pcns = oc.eq(oc.MX(0),1,'<sos>')
#                             subexp[var.nme]  = oc.MX(0)
#
#                             for i,val in enumerate(range(var.lob,var.upb+1)):
#                                 toadd.append(oc.variable(var.nme+'_b'+str(val),0,1,int(var.val==val),True))
#                                 subexp[var.nme]  += val*toadd[-1].sym
#                                 tmp_sos1.group.append(var.nme+'_b'+str(val))
#                                 tmp_sos1.weights.append(val)
#                                 tmp_pcns.exp += toadd[-1].sym
#
#                             if ty == 'x' or ty == 'y':
#                                 newvarlist += [oc.variable('d'+var.nme+'_b'+str(val),-1,1,0)]
#                                 new_constraints[-1].exp += val*newvarlist[-1].sym
#
#                             self.pcns = self.pcns + [tmp_pcns] if self.pcns != [] else [tmp_pcns]
#                             self.sos1 = self.sos1 + [tmp_sos1] if not self.sos1 is None else [tmp_sos1]
#
#
#                         # adapt the constraints
#                         if self.pcns != []:
#                             for i in range(len(self.pcns)): self.pcns[i].exp = oc.substitute(self.pcns[i].exp,var.sym,subexp[var.nme])
#                         if self.icns != []:
#                             for i in range(len(self.icns)): self.icns[i].exp = oc.substitute(self.icns[i].exp,var.sym,subexp[var.nme])
#                         if self.fcns != []:
#                             for i in range(len(self.fcns)): self.fcns[i].exp = oc.substitute(self.fcns[i].exp,var.sym,subexp[var.nme])
#
#
#                         # adapt the dynamics
#                         if self.ode != []:
#                             for i in range(len(self.ode)): self.ode[i] = oc.substitute(self.ode[i],var.sym,subexp[var.nme])
#                         if self.itr != []:
#                             for i in range(len(self.itr)): self.itr[i] = oc.substitute(self.itr[i],var.sym,subexp[var.nme])
#
#                         # adapt the objective
#                         if not self.lag is None: self.lag = oc.substitute(self.lag,var.sym,subexp[var.nme])
#                         if not self.ipn is None: self.ipn = oc.substitute(self.ipn,var.sym,subexp[var.nme])
#                         if not self.may is None: self.may = oc.substitute(self.may,var.sym,subexp[var.nme])
#
#
#                 # collect info to remap the results obtained with the binarized problem into the original problem
#                 remapfnc[ty] = oc.Function(ty,\
#                                            [v.sym for v in toadd],\
#                                            [subexp[getattr(self,ty)[i].nme] for i in torem],\
#                                            [v.nme for v in toadd],\
#                                            [getattr(self,ty)[i].nme for i in torem])
#
#                 # adapt the set of variables
#                 setattr(self,ty,[var for i,var in enumerate(getattr(self,ty)) if not i in torem] + toadd)
#
#
#                 if ty == 'x':
#                     for k in range(len(new_constraints)): new_constraints[k].exp -= self.ode[torem[k]]
#                     self.pcns = self.pcns + new_constraints if self.pcns != [] else new_constraints
#                     self.ode = [self.ode[k] for k in range(len(self.ode)) if not k in torem] + [newvarlist[k].sym for k in range(len(newvarlist))]
#                     self.u + self.v += newvarlist if self.u + self.v != [] else newvarlist
#
#
#                 if ty == 'y':
#                     for k in range(len(new_constraints)): new_constraints[k].exp -= self.itr[k]
#                     self.pcns = self.pcns + new_constraints if self.pcns != [] else new_constraints
#                     self.itr = [self.itr[k] for k in range(len(self.itr)) if not k in torem] + [newvarlist[k].sym for k in range(len(newvarlist))]
#                     self.u + self.v = self.u + self.v + newvarlist if self.u + self.v != [] else newvarlist
#
#         return remapfnc
#
#
# #
#
#
#
#
#     # perform a binary convexification on the problem controls and parameters
#     def binary_convexification(self):
#
#         # function to checks if a list of symbols is contained in another (ignoring order)
#         def contains(symlist1,symlist2):
#             if len(symlist2) == 0:
#                 return True
#             else:
#                 for k,l1 in enumerate(symlist1):
#                     if (symlist2[0] == l1).is_one():
#                         return contains(symlist1[:k]+symlist1[k+1:],symlist2[1:])
#                 return False
#
#
#
#         remap_info = {'name_in':[],'name_out':[],'in':[],'out':[]}
#
#
#
#
#         for ty in ['y','c']:
#             vartorem = []
#             current_values = oc.DM()
#             symlist = []
#
#
#             if getattr(self,ty) != []:
#                 for k,var in enumerate(getattr(self,ty)):
#                     if var.dsc:
#                         symlist.append(var.sym)
#                         vartorem.append(k)
#                         current_values = oc.horzcat(current_values, oc.repmat(var.val,int(oc.ceil(self.t.val.numel()/oc.DM(var.val).numel())))[:self.t.val.numel()-1])
#
#             if len(symlist) > 0:
#
#                 # collect all the possible assignments for the binary variables
#                 assignments = oc.cartesianProduct([[1,0]]*len(symlist))
#
#                 # eliminate the assignments that violate the sos constraints
#                 cnstorem = []
#                 if self.pcns != []:
#                     for k,cns in enumerate(self.pcns):
#                         if '<sos>' in cns.lbl:
#                             deplist = [v for v in oc.symvar(cns.exp)]
#                             if contains(symlist,deplist):
#                                cnstorem.append(k)
#                                tmp_f = oc.Function('tf',symlist,[cns.exp]);
#                                tmp_eval = [tmp_f(*a) for a in assignments]
#                                assignments = [a for k,a in enumerate(assignments) if tmp_eval[k] >= cns.lob and tmp_eval[k] <= cns.upb]
#
#                     # remove the considered sos constraints:
#                     self.pcns = [self.pcns[k] for k in range(len(self.pcns)) if not k in cnstorem]
#
#
#
#                 # represent each assignment with one variable
#                 vartoadd = [oc.variable('bcvx'+str(k+1),0,1,oc.sum2(oc.fabs(current_values - oc.repmat(oc.DM(assignments[k]).T,self.t.val.numel()-1,1)))==0,True) for k in range(len(assignments))]
#
#                 # symbols to substitute
#                 oldsyms = oc.vertcat(*symlist)
#
#
#                 remap_info['name_in'] += [v.nme for v in vartoadd]
#                 remap_info['name_out'] += [s.name() for s in symlist]
#                 remap_info['in'] += [v.sym for v in vartoadd]
#                 remap_info['out'] += [sum([a[i]*vartoadd[k].sym for k,a in enumerate(assignments)]) for i in range(len(symlist))]
#
#
#                 # adapt the variables set
#                 setattr(self,ty,[var for i,var in enumerate(getattr(self,ty)) if not i in vartorem] + vartoadd)
#
#                 # adapt the constraints
#                 if self.icns != []:
#                     for k in range(len(self.icns)):
#                         if oc.which_depends(self.icns[k].exp,oldsyms,1,True)[0]:
#                             tmp_exp = oc.MX(0)
#                             for i,ass in enumerate(assignments):
#                                 tmp_exp += vartoadd[i].sym*oc.substitute(self.icns[k].exp,oldsyms,oc.DM(ass))
#                             self.icns[k].exp = tmp_exp
#                             self.icns[k].lbl += '<semi_cvx>'
#
#                 if self.pcns != []:
#                     for k in range(len(self.pcns)):
#                         if oc.which_depends(self.pcns[k].exp,oldsyms,1,True)[0]:
#                             tmp_exp = oc.MX(0)
#                             for i,ass in enumerate(assignments):
#                                 tmp_exp += vartoadd[i].sym*oc.substitute(self.pcns[k].exp,oldsyms,oc.DM(ass))
#                             self.pcns[k].exp = tmp_exp
#                             self.pcns[k].lbl += '<semi_cvx>'
#
#                 if self.fcns != []:
#                     for k in range(len(self.fcns)):
#                         if oc.which_depends(self.fcns[k].exp,oldsyms,1,True)[0]:
#                             tmp_exp = oc.MX(0)
#                             for i,ass in enumerate(assignments):
#                                 tmp_exp += vartoadd[i].sym*oc.substitute(self.fcns[k].exp,oldsyms,oc.DM(ass))
#                             self.fcns[k].exp = tmp_exp
#                             self.fcns[k].lbl += '<semi_cvx>'
#
#
#                 # adapt the dynamics
#                 if self.ode != []:
#                     for k in range(len(self.ode)):
#                         if oc.which_depends(self.ode[k],oldsyms,1,True)[0]:
#                             tmp_exp = oc.MX(0)
#                             for i,ass in enumerate(assignments):
#                                 tmp_exp += vartoadd[i].sym*oc.substitute(self.ode[k],oldsyms,oc.DM(ass))
#                             self.ode[k] = tmp_exp
#
#                 if self.itr != []:
#                     for k in range(len(self.itr)):
#                         if oc.which_depends(self.itr[k],oldsyms,1,True)[0]:
#                             tmp_exp = oc.MX(0)
#                             for i,ass in enumerate(assignments):
#                                 tmp_exp += vartoadd[i].sym*oc.substitute(self.itr[k],oldsyms,oc.DM(ass))
#                             self.itr[k] = tmp_exp
#
#
#                 # adapt the objective
#                 if not self.lag is None and oc.which_depends(self.lag,oldsyms,1,True)[0]:
#                         tmp_exp = oc.MX(0)
#                         for i,ass in enumerate(assignments):
#                             tmp_exp += vartoadd[i].sym*oc.substitute(self.lag,oldsyms,oc.DM(ass))
#                         self.lag = tmp_exp
#
#                 if not self.ipn is None and oc.which_depends(self.ipn,oldsyms,1,True)[0]:
#                     tmp_exp = oc.MX(0)
#                     for i,ass in enumerate(assignments):
#                         tmp_exp += vartoadd[i].sym*oc.substitute(self.ipn,oldsyms,oc.DM(ass))
#                     self.ipn = tmp_exp
#
#                 if not self.may is None and oc.which_depends(self.may,oldsyms,1,True)[0]:
#                     tmp_exp = oc.MX(0)
#                     for i,ass in enumerate(assignments):
#                         tmp_exp += vartoadd[i].sym*oc.substitute(self.may,oldsyms,oc.DM(ass))
#                     self.ipn = tmp_exp
#
#
#                 # add a new sos constraint
#                 self.pcns.append(oc.eq(sum([v.sym for v in vartoadd]),1,'<sos>'))
#
#                 # return a function that remaps the new variable values into the old ones
#                 return oc.Function('binary_convexification_remap',remap_info['in'],remap_info['out'],remap_info['name_in'],remap_info['name_out'])
