#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:58:12 2017

@author: Massimo De Mauri
"""

import copy
import MIRT_OC as oc


class oc_problem:

    def __init__(self,name):
        self.name = name
        self.t = [] # time vector
        self.i = [] # time varying inputs
        self.x = [] # state
        self.y = [] # algebraic variables
        self.z = [] # finite state (discrte transitions)
        self.s = [] # slacks
        self.p = [] # parameters
        self.c = [] # controls

        self.ode = [] # well ODE
        self.alg = [] # algebraic equation of DAE
        self.dtr = [] # discrete transitions

        self.pcns = []  # path constraints
        self.icns = []  # initial constraints
        self.fcns = []  # final constraints

        self.may = None  # end cost
        self.lag = None  # stage cost
        self.dpn = None  # disctrete stage cost

        self.sos1 = None
        self.wsos1 = None
        self.dsc = None
        self.ord = None



    def copy(self):


       new_model = oc_problem(self.name)
       new_model.t = self.t
       new_model.i = self.i
       new_model.x = self.x
       new_model.y = self.y
       new_model.z = self.z
       new_model.s = self.s
       new_model.c = self.c
       new_model.p = self.p

       new_model.ode = self.ode
       new_model.alg = self.alg
       new_model.dtr = self.dtr

       new_model.pcns = self.pcns
       new_model.icns = self.icns
       new_model.fcns = self.fcns

       new_model.may = self.may
       new_model.lag = self.lag
       new_model.dpn = self.dpn

       return new_model


    def deepcopy(self):
        return copy.deepcopy(self)

    def __str__(self):

        self.classify()
        space = 3*' '
        des = space + 'Optimal control problem:'
        sep = '---------------------------------------------------------------------------------\n'

        out = sep +self.name + '\n'+des+'\n' + sep

        out += 'time:\n' + space + str(self.t) + '\n\n'


        if not self.i == []:
            out += 'ext. inputs:\n'
            for i in self.i:
                    out += space + str(i)+'\n'
            out += '\n'

        out += 'states:\n'
        for i in self.x:
            out += space + str(i)+'\n'
        out += '\n'

        if not self.y == []:
            out += 'alg. vars:\n'
            for i in self.y:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.z == []:
            out += 'discrete transition states:\n'
            for i in self.z:
                out += space + str(i)+'\n'
        out += '\n'

        if not self.s == []:
            out += 'slack variables:\n'
            for i in self.s:
                out += space + str(i)+'\n'
        out += '\n'


        if not self.c == []:
            out += 'controls:\n'
            for i in self.c:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.p == []:
            out += 'parameters:\n'
            for i in self.p:
                out += space + str(i)+'\n'
            out += '\n'

        out += 'diff. eq.:\n'
        for i in range(len(self.ode)):
            out += space + 'dot(' + self.x[i].nme +') = '+ str(self.ode[i])+'\n'
        out +='\n'

        if not self.alg == []:
            out += 'alg. eq.:\n'
            for i in range(len(self.alg)):
                out += space + str(self.alg[i])+' = 0\n'
            out += '\n'

        if not self.dtr == []:
            out += 'discrete transitions:\n'
            for i in range(len(self.dtr)):
                out += space + self.z[i].nme + '[k+1] = ' + str(self.dtr[i])+'[k] + ' +self.z[i].nme + '[k]\n'
            out += '\n'
        if not self.icns == []:
            out += 'initial const.:\n'
            for i in self.icns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.pcns == []:
            out += 'path const.:\n'
            for i in self.pcns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.fcns == []:
            out += 'final const.:\n'
            for i in self.fcns:
                out += space + str(i)+'\n'
            out += '\n'

        if not self.lag is None: out += 'Lagrange\'s obj.: ' + space + str(self.lag)+'\n\n'
        if not self.dpn is None: out += 'Discrete penalty: ' + space + str(self.dpn)+'\n\n'
        if not self.may is None: out += 'Mayer\'s obj.: ' + space + str(self.may)+'\n\n'

        if not self.sos1 is None:
            out += 'SOS constraints:\n'
            for i in range(len(self.sos1)):
                out += space + str(self.sos1[i]) +'\n'


        out += '\n'

        return out



    def classify(self):

        if any([i.dsc for i in self.c]):
            self.dsc = True
        else:
            self.dsc = False

        var_list = []
        if self.x != []: var_list += [v.sym for v in self.x]
        if self.z != []: var_list += [v.sym for v in self.z]
        if self.s != []: var_list += [v.sym for v in self.s]
        if self.c != []: var_list += [v.sym for v in self.c]
        if self.p != []: var_list += [v.sym for v in self.p]
        if self.y != []: var_list += [v.sym for v in self.y]

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

        if self.z != []:
            for k in range(len(self.z)):
                vval_dict[self.z[k].nme] = self.z[k].val

        if self.s != []:
            for k in range(len(self.s)):
                vval_dict[self.s[k].nme] = self.s[k].val

        if self.c != []:
            for k in range(len(self.c)):
                vval_dict[self.c[k].nme] = self.c[k].val

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

        if self.z != []:
            for k in range(len(self.z)):
                if self.z[k].nme in vval_dict:
                    self.z[k].val = vval_dict[self.z[k].nme]

        if self.s != []:
            for k in range(len(self.s)):
                if self.s[k].nme in vval_dict:
                    self.s[k].val = vval_dict[self.s[k].nme]

        if self.c != []:
            for k in range(len(self.c)):
                if self.c[k].nme in vval_dict:
                    self.c[k].val = vval_dict[self.c[k].nme]





    # transform a mixed integer problem into a mixed binary one
    def binarize(self,vartypes = ['p','x','z','s','c']):

        subexp = {}
        remapfnc = {}

        for ty in vartypes:
            if getattr(self,ty) != []:
                torem = []
                toadd = []
                if ty == 'x' or ty == 'z':
                    new_constraints = []
                    newvarlist = []

                for k,var in enumerate(getattr(self,ty)):
                    if var.dsc == True and (var.lob != 0 or var.upb != 1):

                        if var.upb-var.lob == 1:
                            torem.append(k)
                            toadd.append(oc.variable(var.nme+'_b',0,1,var.val-var.lob,True))
                            subexp[var.nme] = toadd[-1].sym + var.lob

                        else:

                            if ty == 'x': new_constraints.append(oc.eq(oc.MX(0),0))
                            if ty == 'z': new_constraints.append(oc.eq(oc.MX(0),0))

                            torem.append(k)
                            tmp_sos1 = oc.sos1_constraint([],[])
                            tmp_pcns = oc.eq(oc.MX(0),1,'<sos>')
                            subexp[var.nme]  = oc.MX(0)



                            for i,val in enumerate(range(var.lob,var.upb+1)):
                                toadd.append(oc.variable(var.nme+'_b'+str(val),0,1,int(var.val==val),True))
                                subexp[var.nme]  += val*toadd[-1].sym
                                tmp_sos1.group.append(var.nme+'_b'+str(val))
                                tmp_sos1.weights.append(val)
                                tmp_pcns.exp += toadd[-1].sym

                            if ty == 'x' or ty == 'z':
                                newvarlist += [oc.variable('d'+var.nme+'_b'+str(val),-1,1,0)]
                                new_constraints[-1].exp += val*newvarlist[-1].sym



                            self.pcns = self.pcns + [tmp_pcns] if self.pcns != [] else [tmp_pcns]
                            self.sos1 = self.sos1 + [tmp_sos1] if not self.sos1 is None else [tmp_sos1]


                        # adapt the constraints
                        if self.pcns != []:
                            for i in range(len(self.pcns)): self.pcns[i].exp = oc.substitute(self.pcns[i].exp,var.sym,subexp[var.nme])
                        if self.icns != []:
                            for i in range(len(self.icns)): self.icns[i].exp = oc.substitute(self.icns[i].exp,var.sym,subexp[var.nme])
                        if self.fcns != []:
                            for i in range(len(self.fcns)): self.fcns[i].exp = oc.substitute(self.fcns[i].exp,var.sym,subexp[var.nme])


                        # adapt the dynamics
                        if self.ode != []:
                            for i in range(len(self.ode)): self.ode[i] = oc.substitute(self.ode[i],var.sym,subexp[var.nme])
                        if self.dtr != []:
                            for i in range(len(self.dtr)): self.dtr[i] = oc.substitute(self.dtr[i],var.sym,subexp[var.nme])

                        # adapt the objective
                        if not self.lag is None: self.lag = oc.substitute(self.lag,var.sym,subexp[var.nme])
                        if not self.dpn is None: self.dpn = oc.substitute(self.dpn,var.sym,subexp[var.nme])
                        if not self.may is None: self.may = oc.substitute(self.may,var.sym,subexp[var.nme])


                # collect info to remap the results obtained with the binarized problem into the original problem
                remapfnc[ty] = oc.Function(ty,\
                                           [v.sym for v in toadd],\
                                           [subexp[getattr(self,ty)[i].nme] for i in torem],\
                                           [v.nme for v in toadd],\
                                           [getattr(self,ty)[i].nme for i in torem])

                # adapt the set of variables
                setattr(self,ty,[var for i,var in enumerate(getattr(self,ty)) if not i in torem] + toadd)


                if ty == 'x':
                    for k in range(len(new_constraints)): new_constraints[k].exp -= self.ode[torem[k]]
                    self.pcns = self.pcns + new_constraints if self.pcns != [] else new_constraints
                    self.ode = [self.ode[k] for k in range(len(self.ode)) if not k in torem] + [newvarlist[k].sym for k in range(len(newvarlist))]
                    self.c += newvarlist if self.c != [] else newvarlist


                if ty == 'z':
                    for k in range(len(new_constraints)): new_constraints[k].exp -= self.dtr[k]
                    self.pcns = self.pcns + new_constraints if self.pcns != [] else new_constraints
                    self.dtr = [self.dtr[k] for k in range(len(self.dtr)) if not k in torem] + [newvarlist[k].sym for k in range(len(newvarlist))]
                    self.c = self.c + newvarlist if self.c != [] else newvarlist

        return remapfnc


#




    # perform a binary convexification on the problem controls and parameters
    def binary_convexification(self):

        # function to checks if a list of symbols is contained in another (ignoring order)
        def contains(symlist1,symlist2):
            if len(symlist2) == 0:
                return True
            else:
                for k,l1 in enumerate(symlist1):
                    if (symlist2[0] == l1).is_one():
                        return contains(symlist1[:k]+symlist1[k+1:],symlist2[1:])
                return False



        remap_info = {'name_in':[],'name_out':[],'in':[],'out':[]}




        for ty in ['z','c']:
            vartorem = []
            current_values = oc.DM()
            symlist = []


            if getattr(self,ty) != []:
                for k,var in enumerate(getattr(self,ty)):
                    if var.dsc:
                        symlist.append(var.sym)
                        vartorem.append(k)
                        current_values = oc.horzcat(current_values, oc.repmat(var.val,int(oc.ceil(self.t.val.numel()/oc.DM(var.val).numel())))[:self.t.val.numel()-1])

            if len(symlist) > 0:

                # collect all the possible assignments for the binary variables
                assignments = oc.cartesianProduct([[1,0]]*len(symlist))

                # eliminate the assignments that violate the sos constraints
                cnstorem = []
                if self.pcns != []:
                    for k,cns in enumerate(self.pcns):
                        if '<sos>' in cns.lbl:
                            deplist = [v for v in oc.symvar(cns.exp)]
                            if contains(symlist,deplist):
                               cnstorem.append(k)
                               tmp_f = oc.Function('tf',symlist,[cns.exp]);
                               tmp_eval = [tmp_f(*a) for a in assignments]
                               assignments = [a for k,a in enumerate(assignments) if tmp_eval[k] >= cns.lob and tmp_eval[k] <= cns.upb]

                    # remove the considered sos constraints:
                    self.pcns = [self.pcns[k] for k in range(len(self.pcns)) if not k in cnstorem]



                # represent each assignment with one variable
                vartoadd = [oc.variable('bcvx'+str(k+1),0,1,oc.sum2(oc.fabs(current_values - oc.repmat(oc.DM(assignments[k]).T,self.t.val.numel()-1,1)))==0,True) for k in range(len(assignments))]

                # symbols to substitute
                oldsyms = oc.vertcat(*symlist)


                remap_info['name_in'] += [v.nme for v in vartoadd]
                remap_info['name_out'] += [s.name() for s in symlist]
                remap_info['in'] += [v.sym for v in vartoadd]
                remap_info['out'] += [sum([a[i]*vartoadd[k].sym for k,a in enumerate(assignments)]) for i in range(len(symlist))]


                # adapt the variables set
                setattr(self,ty,[var for i,var in enumerate(getattr(self,ty)) if not i in vartorem] + vartoadd)

                # adapt the constraints
                if self.icns != []:
                    for k in range(len(self.icns)):
                        if oc.which_depends(self.icns[k].exp,oldsyms,1,True)[0]:
                            tmp_exp = oc.MX(0)
                            for i,ass in enumerate(assignments):
                                tmp_exp += vartoadd[i].sym*oc.substitute(self.icns[k].exp,oldsyms,oc.DM(ass))
                            self.icns[k].exp = tmp_exp
                            self.icns[k].lbl += '<semi_cvx>'

                if self.pcns != []:
                    for k in range(len(self.pcns)):
                        if oc.which_depends(self.pcns[k].exp,oldsyms,1,True)[0]:
                            tmp_exp = oc.MX(0)
                            for i,ass in enumerate(assignments):
                                tmp_exp += vartoadd[i].sym*oc.substitute(self.pcns[k].exp,oldsyms,oc.DM(ass))
                            self.pcns[k].exp = tmp_exp
                            self.pcns[k].lbl += '<semi_cvx>'

                if self.fcns != []:
                    for k in range(len(self.fcns)):
                        if oc.which_depends(self.fcns[k].exp,oldsyms,1,True)[0]:
                            tmp_exp = oc.MX(0)
                            for i,ass in enumerate(assignments):
                                tmp_exp += vartoadd[i].sym*oc.substitute(self.fcns[k].exp,oldsyms,oc.DM(ass))
                            self.fcns[k].exp = tmp_exp
                            self.fcns[k].lbl += '<semi_cvx>'


                # adapt the dynamics
                if self.ode != []:
                    for k in range(len(self.ode)):
                        if oc.which_depends(self.ode[k],oldsyms,1,True)[0]:
                            tmp_exp = oc.MX(0)
                            for i,ass in enumerate(assignments):
                                tmp_exp += vartoadd[i].sym*oc.substitute(self.ode[k],oldsyms,oc.DM(ass))
                            self.ode[k] = tmp_exp

                if self.dtr != []:
                    for k in range(len(self.dtr)):
                        if oc.which_depends(self.dtr[k],oldsyms,1,True)[0]:
                            tmp_exp = oc.MX(0)
                            for i,ass in enumerate(assignments):
                                tmp_exp += vartoadd[i].sym*oc.substitute(self.dtr[k],oldsyms,oc.DM(ass))
                            self.dtr[k] = tmp_exp


                # adapt the objective
                if not self.lag is None and oc.which_depends(self.lag,oldsyms,1,True)[0]:
                        tmp_exp = oc.MX(0)
                        for i,ass in enumerate(assignments):
                            tmp_exp += vartoadd[i].sym*oc.substitute(self.lag,oldsyms,oc.DM(ass))
                        self.lag = tmp_exp

                if not self.dpn is None and oc.which_depends(self.dpn,oldsyms,1,True)[0]:
                    tmp_exp = oc.MX(0)
                    for i,ass in enumerate(assignments):
                        tmp_exp += vartoadd[i].sym*oc.substitute(self.dpn,oldsyms,oc.DM(ass))
                    self.dpn = tmp_exp

                if not self.may is None and oc.which_depends(self.may,oldsyms,1,True)[0]:
                    tmp_exp = oc.MX(0)
                    for i,ass in enumerate(assignments):
                        tmp_exp += vartoadd[i].sym*oc.substitute(self.may,oldsyms,oc.DM(ass))
                    self.dpn = tmp_exp


                # add a new sos constraint
                self.pcns.append(oc.eq(sum([v.sym for v in vartoadd]),1,'<sos>'))

                # return a function that remaps the new variable values into the old ones
                return oc.Function('binary_convexification_remap',remap_info['in'],remap_info['out'],remap_info['name_in'],remap_info['name_out'])




    def epigraph_lag_obj(self, minimum = -oc.DM.inf(), maximum = oc.DM.inf(), integration_opts = None):

        if not self.lag is None:

            # define integrator
            dt = oc.input('dt',oc.vertcat(self.t.val[1:]-self.t.val[:-1],1))
            x0 = oc.MX.sym('x0')

            # collect the symbols
            psyms = oc.vertcat(*[v.sym for v in self.p]) if self.p != [] else oc.MX()
            xsyms = oc.vertcat(*[v.sym for v in self.x]) if self.x != [] else oc.MX()
            zsyms = oc.vertcat(*[v.sym for v in self.z]) if self.z != [] else oc.MX()
            ssyms = oc.vertcat(*[v.sym for v in self.s]) if self.s != [] else oc.MX()
            csyms = oc.vertcat(*[v.sym for v in self.c]) if self.c != [] else oc.MX()
            isyms = oc.vertcat(*[v.sym for v in self.i]) if self.i != [] else oc.MX()

            # evaluate integration (symbolically)
            intg = oc.integrate_ode(x0,oc.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms,self.t.sym),[self.lag],dt.sym,integration_opts)
            tmp_exp = intg.call({'x0':0,'p':oc.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms,self.t.sym,dt.sym)})['xf']

            # modify the model accordingly
            self.i      = self.i + [dt]                                         if self.i != []       else [dt]
            self.c      = self.c + [oc.variable('lag_obj',minimum,maximum)]                 if self.c != []       else [oc.variable('lag_obj',minimum,maximum)]
            self.pcns   = self.pcns + [oc.leq(tmp_exp,self.c[-1].sym)]     if self.pcns != []    else [oc.leq(tmp_exp,self.c[-1].sym,lbl='<obj_lift>')]
            self.dpn    = self.dpn + self.c[-1].sym                                         if not self.dpn is None     else self.c[-1].sym
            self.lag    = 0.0


    def epigraph_dpn_obj(self, minimum = -oc.DM.inf(), maximum = oc.DM.inf()):

        if not self.dpn is None:

            # modify the model accordingly
            self.c      = self.c + [oc.variable('dpn_obj',minimum,maximum)] if self.c != []       else [oc.variable('lag_obj',minimum,maximum)]
            self.pcns   = self.pcns + [oc.leq(self.dpn,self.c[-1].sym)]     if self.pcns != []    else [oc.leq(self.dpn,self.c[-1].sym,lbl='<obj_lift>')]
            self.dpn    = self.c[-1].sym


    def epigraph_may_obj(self, minimum = -oc.DM.inf(), maximum = oc.DM.inf()):

        if not self.may is None:
            self.p    = self.p + [oc.variable('may_obj',minimum,maximum)] if self.p != []     else [oc.variable('may_obj',minimum,maximum)]
            self.fcns = self.fcns + [oc.leq(self.may,self.p[-1].sym)]     if self.fcns != []  else [oc.leq(self.may,self.p[-1].sym)]
            self.may  = self.p[-1].sym


    def epigraph_obj(self, options = None):

        opts = {'min_lag': -oc.DM.inf(),\
                'min_dpn': -oc.DM.inf(),\
                'min_may': -oc.DM.inf(),\
                'max_lag':  oc.DM.inf(),\
                'max_dpn':  oc.DM.inf(),\
                'max_may':  oc.DM.inf(),\
                'integration_options': None
                }

        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('Option not recognized' + k)

        if not self.may is None and oc.get_order(self.may) > 1: self.epigraph_may_obj(opts['min_may'],opts['max_may'])
        if not self.dpn is None and oc.get_order(self.dpn) > 1: self.epigraph_dpn_obj(opts['min_dpn'],opts['max_dpn'])
        if not self.lag is None and oc.get_order(self.lag) > 1: self.epigraph_lag_obj(opts['min_lag'],opts['max_lag'],opts['integration_options'])
