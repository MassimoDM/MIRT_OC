#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:28:19 2017

@author: Massimo De Mauri
"""
import MIRT_OC as oc
import casadi as cs
from warnings import warn

def multipleShooting(model,problem,options=None,unpack_objective=False):

    opts = {}
    opts['integration_opts'] = None
    opts['leq0_only'] = False
    opts['with_initial_cnss'] = True
    opts['with_final_cnss'] = True

    if not options is None:
        for k in options.keys():
            if k in opts:
                opts[k] = options[k]
            else:
                warn('Option not recognized : ' + k)

    nSteps = model.t.val.numel()-1

    # collect the parameters
    if model.p != []:
        pnum = len(model.p)
        pindxs = list(range(pnum))
        psyms = cs.vertcat(*[v.sym for v in model.p])

        pdata = []
        for k in range(pnum):
            pdata.append({'lob': cs.repmat(cs.reshape(model.p[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.p[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.p[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.p[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.p[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.p[k].val).numel())))[:nSteps+1],\
                          'dsc': model.p[k].dsc,\
                          'lbl': model.p[k].lbl+"<p>",\
                          'nme': model.p[k].nme
                          })

        plbl = ['<ocp_p>']*pnum
        for i in range(pnum):
            if not model.p[i].lbl is None: plbl[i] += model.p[i].lbl

    else:
        pdata = []
        pnum = 0
        pindxs = []
        psyms = cs.MX()
        plbl = []




    # collect the states
    if model.x != []:
        xnum = len(model.x)
        xindxs = list(range(pnum,pnum+xnum))
        xsyms = cs.vertcat(*[v.sym for v in model.x])

        xdata = []
        for k in range(xnum):
            xdata.append({'lob': cs.repmat(cs.reshape(model.x[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.x[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.x[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.x[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.x[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.x[k].val).numel())))[:nSteps+1],\
                          'dsc': model.x[k].dsc,\
                          'lbl': model.x[k].lbl+"<x>",\
                          'nme': model.x[k].nme
                          })

        xlbl = ['<ocp_x>']*xnum
        for i in range(xnum):
            if not model.x[i].lbl is None: xlbl[i] += model.x[i].lbl

    else:
        xdata = []
        xnum = 0
        xindxs = []
        xsyms = cs.MX()
        xlbl = []



    # collect the dicrete transition states
    if model.z != []:
        znum = len(model.z)
        zindxs = list(range(pnum+xnum,pnum+xnum+znum))
        zsyms = cs.vertcat(*[v.sym for v in model.z])

        zdata = []
        for k in range(znum):
            zdata.append({'lob': cs.repmat(cs.reshape(model.z[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.z[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.z[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.z[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.z[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.z[k].val).numel())))[:nSteps+1],\
                          'dsc': model.z[k].dsc,\
                          'lbl': model.z[k].lbl+"<z>",\
                          'nme': model.z[k].nme
                          })

        zlbl = ['<ocp_z>']*znum
        for i in range(znum):
            if not model.z[i].lbl is None: zlbl[i] += model.z[i].lbl

    else:
        zdata = []
        znum = 0
        zindxs = []
        zsyms = cs.MX()
        zlbl = []



    # collect the slack variables
    if model.s != []:
        snum = len(model.s)
        sindxs = list(range(pnum+xnum+znum,pnum+xnum+znum+snum))
        ssyms = cs.vertcat(*[v.sym for v in model.s])

        sdata = []
        for k in range(snum):
            sdata.append({'lob': cs.repmat(cs.reshape(model.s[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.s[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.s[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.s[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.s[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.s[k].val).numel())))[:nSteps+1],\
                          'dsc': model.s[k].dsc,\
                          'lbl': model.s[k].lbl+"<s>",\
                          'nme': model.s[k].nme
                          })

        slbl = ['<ocp_s>']*snum
        for i in range(snum):
            if not model.s[i].lbl is None: slbl[i] += model.s[i].lbl

    else:
        sdata = []
        snum = 0
        sindxs = []
        ssyms = cs.MX()
        slbl = []



    # collect the controls
    if model.c != []:
        cnum = len(model.c)
        cindxs = list(range(pnum+xnum+znum+snum,pnum+xnum+znum+snum+cnum))
        csyms = cs.vertcat(*[v.sym for v in model.c])

        cdata = []
        for k in range(cnum):
            cdata.append({'lob': cs.repmat(cs.reshape(model.c[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.c[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.c[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.c[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.c[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.c[k].val).numel())))[:nSteps+1],\
                          'dsc': model.c[k].dsc,\
                          'lbl': model.c[k].lbl+"<c>",\
                          'nme': model.c[k].nme
                          })

        clbl = ['<ocp_c>']*cnum
        for i in range(cnum):
            if not model.c[i].lbl is None: clbl[i] += model.c[i].lbl

    else:
        cdata = []
        cnum = 0
        cindxs = []
        csyms = []
        clbl = []


    # collect the external inputs
    if model.i != []:
        isyms = cs.vertcat(*[v.sym for v in model.i])
        idata = cs.vertcat(*[v.val.T for v in model.i])
    else:
        isyms = cs.DM()
        idata = cs.DM(0,nSteps+1)



    # collect the algebraic states
    if model.y != []:
        ynum = len(model.y)
        ysyms = cs.vertcat(*[v.sym for v in model.y])

    else:
        ynum = 0
        ysyms = cs.MX()


    # create the set of variables for multiple shooting
    syms = cs.vertcat(psyms,xsyms,zsyms,ssyms,csyms)
    vardata = pdata+xdata+zdata+sdata+cdata
    numVars = len(vardata)

    # construct the problem variables
    problem.var['sym'] = cs.MX.sym('V',nSteps*numVars + snum + znum + xnum + pnum,1)
    problem.var['lob'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['lob'] for k in range(numVars)]),-1,1)[:nSteps*numVars + snum + znum + xnum + pnum])
    problem.var['upb'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['upb'] for k in range(numVars)]),-1,1)[:nSteps*numVars + snum + znum + xnum + pnum])
    problem.var['val'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['val'] for k in range(numVars)]),-1,1)[:nSteps*numVars + snum + znum + xnum + pnum])
    problem.var['dsc'] = [v['dsc'] for v in pdata+xdata+zdata+sdata+cdata]*nSteps + [v['dsc'] for v in pdata+xdata+zdata+sdata]
    problem.var['nme'] = [v['nme'] for v in pdata+xdata+zdata+sdata+cdata]*nSteps + [v['nme'] for v in pdata+xdata+zdata+sdata]
    problem.var['lbl'] = [v['lbl'] for v in pdata+xdata+zdata+sdata+cdata]*nSteps + [v['lbl'] for v in pdata+xdata+zdata+sdata]
    problem.var['lam'] = None

    # collect data for multiple shooting
    shooting = {}
    shooting['t']  = model.t.val.T
    shooting['dt'] = shooting['t'][1:]-shooting['t'][:-1]
    shooting['i']  = idata


    shooting['p']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in pindxs]],(pnum,nSteps+1))
    shooting['x']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in xindxs]],(xnum,nSteps+1))
    shooting['z']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in zindxs]],(znum,nSteps+1))
    shooting['s']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in sindxs]],(snum,nSteps+1))
    shooting['c']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps)    for j in cindxs]],(cnum,nSteps))


    # collect the constraints
    labels_tmp = []
    problem.cns = {'exp':cs.MX(),'lob':cs.DM(),'upb':cs.DM(),'lbl':[],'lam':None}

    # parameters are treated like states with zero dynamics
    if model.p != []:
        # add constraints
        problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['p'][:,:-1]-shooting['p'][:,1:])
        problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.DM.zeros(pnum,nSteps))
        problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(pnum,nSteps))

        # collect shooting labels
        labels_tmp += ['<ocp_par'+str(i)+'>' for i in range(pnum)]

    # dynamic constraints
    if model.ode != []:
        # define integrator
        dt = cs.MX.sym('dt')
        intg = oc.integrate_ode(xsyms,cs.vertcat(zsyms,ssyms,psyms,isyms,csyms),model.ode,dt,opts['integration_opts'])

        # evaluate integration
        intg = intg.map(nSteps)
        intgval = intg.call({'x0':shooting['x'][:,:-1],'p':cs.vertcat(shooting['z'][:,:-1],shooting['s'][:,:-1],
                                                                      shooting['p'][:,:-1],shooting['i'][:,:-1],
                                                                      shooting['c'],shooting['dt'])})


        # add integration constraints
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],intgval['xf']-shooting['x'][:,1:])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.DM.zeros(xnum,nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(xnum,nSteps))

            # classify the resulting constraints
            tmp_order_a = oc.get_order(cs.vertcat(*model.ode)) # global order
            tmp_order_c = oc.get_order(cs.vertcat(*model.ode),[syms[i] for i in range(syms.numel()) if vardata[i]['dsc'] == 0 ]) # order considering the continuous variables only
            labels_tmp += ['<ocp_int'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(xnum)]

        else:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],intgval['xf']-shooting['x'][:,1:],-intgval['xf']+shooting['x'][:,1:])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(2*xnum,nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(2*xnum,nSteps))

            # classify the resulting constraints
            tmp_order_a = oc.get_order(cs.vertcat(*model.ode)) # global order
            tmp_order_c = oc.get_order(cs.vertcat(*model.ode),[syms[i] for i in range(syms.numel()) if vardata[i]['dsc'] == 0 ]) # order considering the continuous variables only
            labels_tmp += 2*['<ocp_int'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(xnum)]


    # constraints for discrete transitions
    if model.dtr != []:
        # define discrete transition
        dtr = cs.Function('dtr',[cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms)],[cs.vertcat(*model.dtr)])

        # evaluate the discrete transitions
        dtr = dtr.map(nSteps)
        trans = dtr(cs.vertcat(shooting['x'][:,:-1],shooting['z'][:,:-1],shooting['s'][:,:-1],shooting['p'][:,:-1],shooting['i'][:,:-1],shooting['c']))

        # add discrete transition constraints
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['z'][:,:-1]+trans-shooting['z'][:,1:])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.DM.zeros(trans.size()))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(trans.size()))


            # classify the resulting constraints
            tmp_order_a = oc.get_order(cs.vertcat(*model.dtr)) # global order
            tmp_order_c = oc.get_order(cs.vertcat(*model.dtr),[syms[i] for i in range(syms.numel()) if vardata[i]['dsc'] == 0 ]) # order considering the continuous variables only
            labels_tmp += ['<ocp_dst'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(znum)]

        else:

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['z'][:,:-1]+trans-shooting['z'][:,1:],-shooting['z'][:,:-1]-trans+shooting['z'][:,1:])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(trans.shape[0]*2,trans.shape[1]))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(trans.shape[0]*2,trans.shape[1]))


             # classify the resulting constraints
            tmp_order_a = oc.get_order(cs.vertcat(*model.dtr)) # global order
            tmp_order_c = oc.get_order(cs.vertcat(*model.dtr),[syms[i] for i in range(syms.numel()) if vardata[i]['dsc'] == 0 ]) # order considering the continuous variables only
            labels_tmp += 2*['<ocp_dst'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(znum)]




    # add path constraints
    if model.pcns != []:


        if not opts['leq0_only']:
            pcnsf = cs.Function('pcns',[cs.vertcat(model.t.sym,xsyms,zsyms,ssyms,psyms,isyms,csyms)],[cs.vertcat(*[c.exp for c in model.pcns])])
            pcnsf =  pcnsf.map(nSteps)

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],pcnsf(cs.vertcat(shooting['t'][:-1],shooting['x'][:,:-1],
                                                                                shooting['z'][:,:-1],shooting['s'][:,:-1],
                                                                                shooting['p'][:,:-1],shooting['i'][:,:-1],
                                                                                shooting['c'])))
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.repmat(cs.vertcat(*[c.lob for c in model.pcns]),(1,nSteps)))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.repmat(cs.vertcat(*[c.upb for c in model.pcns]),(1,nSteps)))

            labels_tmp += [model.pcns[i].lbl+'<ocp_pc'+str(i)+'>' for i in range(len(model.pcns))]
        else:
            pcnsexp = cs.vertcat(*([c.lob - c.exp for c in model.pcns if c.lob > -cs.DM.inf()] +\
                                   [c.exp - c.upb for c in model.pcns if c.upb <  cs.DM.inf()]))

            pcnsf = cs.Function('pcns',[cs.vertcat(model.t.sym,xsyms,zsyms,ssyms,psyms,isyms,csyms)],[pcnsexp])
            pcnsf =  pcnsf.map(nSteps)

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],pcnsf(cs.vertcat(shooting['t'][:-1],shooting['x'][:,:-1],
                                                                                shooting['z'][:,:-1],shooting['s'][:,:-1],
                                                                                shooting['p'][:,:-1],shooting['i'][:,:-1],
                                                                                shooting['c'])))
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(pcnsexp.shape[0],nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(pcnsexp.shape[0],nSteps))

            labels_tmp += [model.pcns[i].lbl + '<ocp_pc'+str(i)+'>' for i,c in enumerate(model.pcns) if c.lob > -cs.DM.inf()] +\
                          [model.pcns[i].lbl + '<ocp_pc'+str(i)+'>' for i,c in enumerate(model.pcns) if c.upb <  cs.DM.inf()]




    # reshape the constrants vector and add labels
    problem.cns['exp'] = cs.reshape(problem.cns['exp'],(-1,1))
    problem.cns['lob'] = cs.reshape(problem.cns['lob'],(-1,1))
    problem.cns['upb'] = cs.reshape(problem.cns['upb'],(-1,1))
    problem.cns['lbl'] = labels_tmp*nSteps



    # add initial constraints
    if model.icns != [] and opts['with_initial_cnss']:
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(cs.substitute(cs.vertcat(*[c.exp for c in model.icns]),\
                                                    cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms),\
                                                    cs.vertcat(shooting['x'][:,0],shooting['z'][:,0],shooting['s'][:,0],shooting['p'][:,0],shooting['i'][:,0],shooting['c'][:,0])),\
                                      problem.cns['exp'])

            problem.cns['lob'] = cs.vertcat(cs.vertcat(*[c.lob for c in model.icns]),problem.cns['lob'])
            problem.cns['upb'] = cs.vertcat(cs.vertcat(*[c.upb for c in model.icns]),problem.cns['upb'])
            problem.cns['lbl'] = [model.icns[i].lbl+'<ocp_ic'+str(i)+'>' for i in range(len(model.icns))] + problem.cns['lbl']
        else:
            icnsexp = cs.vertcat(*([c.lob - c.exp for c in model.icns if c.lob > -cs.DM.inf()]+\
                                   [c.exp - c.upb for c in model.icns if c.upb <  cs.DM.inf()]))

            problem.cns['exp'] = cs.vertcat(cs.substitute(icnsexp,\
                                                    cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms),\
                                                    cs.vertcat(shooting['x'][:,0],shooting['z'][:,0],shooting['s'][:,0],shooting['p'][:,0],shooting['i'][:,0],shooting['c'][:,0])),\
                                      problem.cns['exp'])

            problem.cns['lob'] = cs.vertcat( -cs.DM.inf(icnsexp.shape[0],1),problem.cns['lob'])
            problem.cns['upb'] = cs.vertcat(cs.DM.zeros(icnsexp.shape[0],1),problem.cns['upb'])
            problem.cns['lbl'] = [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.icns) if c.lob > -cs.DM.inf()] +\
                           [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.icns) if c.upb <  cs.DM.inf()] +\
                           problem.cns['lbl']




    # add final constraints
    if model.fcns != [] and opts['with_final_cnss']:
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],\
                                      cs.substitute(cs.vertcat(*[c.exp for c in model.fcns]),\
                                                    cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms),\
                                                    cs.vertcat(shooting['x'][:,-1],shooting['z'][:,-1],shooting['s'][:,-1],shooting['p'][:,-1],shooting['i'][:,-1])))


            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.vertcat(*[c.lob for c in model.fcns]))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.vertcat(*[c.upb for c in model.fcns]))
            problem.cns['lbl'] = problem.cns['lbl'] + [model.fcns[i].lbl+'<ocp_fc'+str(i)+'>' for i in range(len(model.fcns))]

        else:
            fcnsexp = cs.vertcat(*([c.lob - c.exp for c in model.fcns if c.lob > -cs.DM.inf()]+\
                                   [c.exp - c.upb for c in model.fcns if c.upb <  cs.DM.inf()]))

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],\
                                      cs.substitute(fcnsexp,\
                                                    cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms),\
                                                    cs.vertcat(shooting['x'][:,-1],shooting['z'][:,-1],shooting['s'][:,-1],shooting['p'][:,-1],shooting['i'][:,-1],shooting['c'][:,-1])))

            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(fcnsexp.shape[0],1))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(fcnsexp.shape[0],1))
            problem.cns['lbl'] = problem.cns['lbl'] +\
                            [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.fcns) if c.lob > -cs.DM.inf()] +\
                            [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.fcns) if c.upb <  cs.DM.inf()]



    ## Construct the objective
    problem.obj = {'exp':None,'val':None}

    # add lagrangian objective
    if not model.lag is None:
        # define integrator
        dt = cs.MX.sym('dt')
        l0 = cs.MX.sym('l0')
        intg = oc.integrate_ode(l0,cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,csyms,model.t.sym),[model.lag],dt,opts['integration_opts']).map(nSteps)

        # evaluate integration
        tmp = intg.call({'x0':cs.DM.zeros(nSteps).T,\
                         'p':cs.vertcat(shooting['x'][:,:-1],shooting['z'][:,:-1],shooting['s'][:,:-1],shooting['p'][:,:-1],shooting['i'][:,:-1],shooting['c'],shooting['t'][:-1],shooting['dt'])})['xf']

        if unpack_objective:
            problem.obj['exp'] = tmp.T
        else:
            problem.obj['exp'] = oc.sum2(tmp)


    # add discrete penalties
    if not model.dpn is None:
        dpnf = cs.Function('dpn',[cs.vertcat(model.t.sym,xsyms,zsyms,ssyms,psyms,isyms,csyms)],[model.dpn]).map(nSteps)
        tmp = dpnf(cs.vertcat(shooting['t'][:-1],shooting['x'][:,:-1],shooting['z'][:,:-1],shooting['s'][:,:-1],shooting['p'][:,:-1],shooting['i'][:,:-1],shooting['c']))


        if not problem.obj['exp'] is None:
            if unpack_objective:
                problem.obj['exp'] += tmp.T
            else:
                problem.obj['exp'] += oc.sum2(tmp)
        else:
            if unpack_objective:
                problem.obj['exp'] = tmp.T
            else:
                problem.obj['exp'] = oc.sum2(tmp)

    # add mayer objective
    if not model.may is None:

        tmp = cs.vertcat(problem.obj['exp'],cs.substitute(model.may,\
                                                          cs.vertcat(model.t.sym,xsyms,zsyms,ssyms,psyms,isyms),\
                                                          cs.vertcat(shooting['t'][:,-1],shooting['x'][:,-1],shooting['z'][:,-1],shooting['s'][:,-1],shooting['p'][:,-1],shooting['i'][:,-1])))
        tmp = cs.substitute(model.may,\
                            cs.vertcat(model.t.sym,xsyms,zsyms,ssyms,psyms,isyms),\
                            cs.vertcat(shooting['t'][:,-1],shooting['x'][:,-1],shooting['z'][:,-1],shooting['s'][:,-1],shooting['p'][:,-1],shooting['i'][:,-1]))

        if not problem.obj['exp'] is None:
            if unpack_objective:
                problem.obj['exp'][-1] += tmp
            else:
                problem.obj['exp'] += tmp
        else:
            problem.obj['exp'] = cs.vertcat(cs.MX.zeros(),tmp)







    # add sos1 constraints
    if not model.sos1 is None:
        problem.sos1 = []
        varsname = [v['nme'] for v in vardata]
        for c in range(len(model.sos1)):
            indxs = [i for i in range(len(varsname)) if varsname[i] in model.sos1[c].group]
            if model.sos1[c].weights != []:
                problem.sos1.append({'g': [[t*numVars + i for i in indxs] for t in range(nSteps)],\
                                     'w': [cs.substitute(model.sos1[c].weights,cs.vertcat(xsyms,zsyms,ssyms,psyms,isyms,model.t.sym),cs.vertcat(shooting['x'][:,-1],shooting['z'][:,-1],shooting['s'][:,-1],shooting['p'][:,-1],shooting['i'][:,-1],shooting['t'][:,-1])) for t in range(nSteps)]})
            else:
                problem.sos1.append({'g': [[t*numVars + i for i in indxs] for t in range(nSteps)],'w':[None]*nSteps})




    return
