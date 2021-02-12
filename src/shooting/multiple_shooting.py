#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:28:19 2017

@author: Massimo De Mauri
"""
from MIRT_OC import get_order, integrate_ode
import casadi as cs
from warnings import warn

def multiple_shooting(model,problem,time_vector,options=None,unpack_objective=False):

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

    nSteps = time_vector.numel()-1
    tsym = model.t
    dtsym = model.dt


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
    if model.y != []:
        ynum = len(model.y)
        yindxs = list(range(pnum+xnum,pnum+xnum+ynum))
        ysyms = cs.vertcat(*[v.sym for v in model.y])

        ydata = []
        for k in range(ynum):
            ydata.append({'lob': cs.repmat(cs.reshape(model.y[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.y[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.y[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.y[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.y[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.y[k].val).numel())))[:nSteps+1],\
                          'lbl': model.y[k].lbl+"<y>",\
                          'nme': model.y[k].nme
                          })

        ylbl = ['<ocp_y>']*ynum
        for i in range(ynum):
            if not model.y[i].lbl is None: ylbl[i] += model.y[i].lbl

    else:
        ydata = []
        ynum = 0
        yindxs = []
        ysyms = cs.MX()
        ylbl = []



    # collect the slack variables
    if model.a != []:
        anum = len(model.a)
        aindxs = list(range(pnum+xnum+ynum,pnum+xnum+ynum+anum))
        asyms = cs.vertcat(*[v.sym for v in model.a])

        adata = []
        for k in range(anum):
            adata.append({'lob': cs.repmat(cs.reshape(model.a[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.a[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.a[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.a[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.a[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.a[k].val).numel())))[:nSteps+1],\
                          'lbl': model.a[k].lbl+"<a>",\
                          'nme': model.a[k].nme
                          })

        albl = ['<ocp_a>']*anum
        for i in range(anum):
            if not model.a[i].lbl is None: albl[i] += model.a[i].lbl

    else:
        adata = []
        anum = 0
        aindxs = []
        asyms = cs.MX()
        albl = []



    # collect the continuous controls
    if model.u != []:
        unum = len(model.u)
        uindxs = list(range(pnum+xnum+ynum+anum,pnum+xnum+ynum+anum+unum))
        usyms = cs.vertcat(*[v.sym for v in model.u])

        udata = []
        for k in range(unum):
            udata.append({'lob': cs.repmat(cs.reshape(model.u[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.u[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.u[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.u[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.u[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.u[k].val).numel())))[:nSteps+1],\
                          'lbl': model.u[k].lbl+"<u>",\
                          'nme': model.u[k].nme
                          })

        ulbl = ['<ocp_u>']*unum
        for i in range(unum):
            if not model.u[i].lbl is None: ulbl[i] += model.u[i].lbl

    else:
        udata = []
        unum = 0
        uindxs = []
        usyms = []
        ulbl = []


    # collect discrete controls
    if model.v != []:
        vnum = len(model.v)
        vindxs = list(range(pnum+xnum+ynum+anum+unum,pnum+xnum+ynum+anum+unum+vnum))
        vsyms = cs.vertcat(*[v.sym for v in model.v])

        vdata = []
        for k in range(vnum):
            vdata.append({'lob': cs.repmat(cs.reshape(model.v[k].lob,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.v[k].lob).numel())))[:nSteps+1],\
                          'upb': cs.repmat(cs.reshape(model.v[k].upb,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.v[k].upb).numel())))[:nSteps+1],\
                          'val': cs.repmat(cs.reshape(model.v[k].val,1,-1),1,int(cs.ceil((nSteps+1)/cs.DM(model.v[k].val).numel())))[:nSteps+1],\
                          'lbl': model.v[k].lbl+"<v>",\
                          'nme': model.v[k].nme
                          })

        vlbl = ['<ocp_v>']*vnum
        for i in range(vnum):
            if not model.v[i].lbl is None: vlbl[i] += model.v[i].lbl

    else:
        vdata = []
        vnum = 0
        vindxs = []
        vsyms = []
        vlbl = []



    # collect the external inputs
    if model.i != []:
        isyms = cs.vertcat(*[v.sym for v in model.i])
        idata = cs.vertcat(*[v.val.T for v in model.i])
    else:
        isyms = cs.DM()
        idata = cs.DM(0,nSteps+1)


    # create the set of variables for multiple shooting
    syms = cs.vertcat(psyms,xsyms,ysyms,asyms,usyms,vsyms)
    vardata = pdata+xdata+ydata+adata+udata+vdata
    numVars = len(vardata)

    # construct the problem variables
    problem.var['sym'] = cs.MX.sym('V',nSteps*numVars + pnum + xnum + ynum + anum,1)
    problem.var['lob'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['lob'] for k in range(numVars)]),-1,1)[:nSteps*numVars + anum + ynum + xnum + pnum])
    problem.var['upb'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['upb'] for k in range(numVars)]),-1,1)[:nSteps*numVars + anum + ynum + xnum + pnum])
    problem.var['val'] = cs.DM(cs.reshape(cs.vertcat(*[vardata[k]['val'] for k in range(numVars)]),-1,1)[:nSteps*numVars + anum + ynum + xnum + pnum])
    problem.var['dsc'] = ([False]*(pnum+xnum+ynum+anum+unum) + [True]*vnum)*nSteps + [False]*(pnum+xnum+ynum+anum)
    problem.var['nme'] = [v['nme'] for v in pdata+xdata+ydata+adata+udata+vdata]*nSteps + [v['nme'] for v in pdata+xdata+ydata+adata]
    problem.var['lbl'] = [v['lbl'] for v in pdata+xdata+ydata+adata+udata+vdata]*nSteps + [v['lbl'] for v in pdata+xdata+ydata+adata]
    problem.var['lam'] = None

    # collect data for multiple shooting
    shooting = {}
    shooting['t']  = time_vector.T
    shooting['dt'] = shooting['t'][1:]-shooting['t'][:-1]
    shooting['i']  = idata


    shooting['p']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in pindxs]],(pnum,nSteps+1))
    shooting['x']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in xindxs]],(xnum,nSteps+1))
    shooting['y']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in yindxs]],(ynum,nSteps+1))
    shooting['a']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps+1)  for j in aindxs]],(anum,nSteps+1))
    shooting['u']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps)    for j in uindxs]],(unum,nSteps))
    shooting['v']  = cs.reshape(problem.var['sym'][[numVars*i+j   for i in range(nSteps)    for j in vindxs]],(vnum,nSteps))

    # collect the constraints
    labels_tmp = []
    problem.cns = {'exp':cs.MX(),'lob':cs.DM(),'upb':cs.DM(),'lbl':[],'lam':None}

    # add path constraints
    if model.pcns != []:

        if not opts['leq0_only']:
            pcnsf = cs.Function('pcns',[cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms,usyms,vsyms,dtsym,)],[cs.vertcat(*[c.exp for c in model.pcns])])
            pcnsf =  pcnsf.map(nSteps)

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],pcnsf(cs.vertcat(shooting['t'][:-1],
                                                                                shooting['x'][:,:-1],
                                                                                shooting['y'][:,:-1],
                                                                                shooting['a'][:,:-1],
                                                                                shooting['p'][:,:-1],
                                                                                shooting['i'][:,:-1],
                                                                                shooting['u'],
                                                                                shooting['v'],
                                                                                shooting['dt'])))

            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.repmat(cs.vertcat(*[c.lob for c in model.pcns]),(1,nSteps)))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.repmat(cs.vertcat(*[c.upb for c in model.pcns]),(1,nSteps)))

            labels_tmp += [model.pcns[i].lbl+'<ocp_pc'+str(i)+'>' for i in range(len(model.pcns))]
        else:
            pcnsexp = cs.vertcat(*([c.lob - c.exp for c in model.pcns if c.lob > -cs.DM.inf()] +\
                                   [c.exp - c.upb for c in model.pcns if c.upb <  cs.DM.inf()]))

            pcnsf = cs.Function('pcns',[cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms,usyms,vsyms,dtsym)],[pcnsexp])
            pcnsf =  pcnsf.map(nSteps)

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],pcnsf(cs.vertcat(shooting['t'][:-1],
                                                                                shooting['x'][:,:-1],
                                                                                shooting['y'][:,:-1],
                                                                                shooting['a'][:,:-1],
                                                                                shooting['p'][:,:-1],
                                                                                shooting['i'][:,:-1],
                                                                                shooting['u'],
                                                                                shooting['v'],
                                                                                shooting['dt'])))
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(pcnsexp.shape[0],nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(pcnsexp.shape[0],nSteps))

            labels_tmp += [model.pcns[i].lbl + '<ocp_pc'+str(i)+'>' for i,c in enumerate(model.pcns) if c.lob > -cs.DM.inf()] +\
                          [model.pcns[i].lbl + '<ocp_pc'+str(i)+'>' for i,c in enumerate(model.pcns) if c.upb <  cs.DM.inf()]


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
        intg = integrate_ode(xsyms,cs.vertcat(tsym,ysyms,asyms,psyms,isyms,usyms,vsyms),model.ode,dtsym,opts['integration_opts'])

        # evaluate integration
        intg = intg.map(nSteps)
        intgval = intg.call({'x0':shooting['x'][:,:-1],'p':cs.vertcat(shooting['t'][:-1],
                                                                      shooting['y'][:,:-1],
                                                                      shooting['a'][:,:-1],
                                                                      shooting['p'][:,:-1],
                                                                      shooting['i'][:,:-1],
                                                                      shooting['u'],
                                                                      shooting['v'],
                                                                      shooting['dt'])})


        # add integration constraints
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['x'][:,1:]-intgval['xf'])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.DM.zeros(xnum,nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(xnum,nSteps))

            # classify the resulting constraints
            tmp_order_a = get_order(cs.vertcat(*model.ode),[syms[i] for i in range(pnum+xnum+ynum+anum+unum+vnum)]) # global order
            tmp_order_c = get_order(cs.vertcat(*model.ode),[syms[i] for i in range(pnum+xnum+ynum+anum+unum)]) # order considering the continuous variables only
            labels_tmp += ['<ocp_int'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(xnum)]

        else:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['x'][:,1:]-intgval['xf'],intgval['xf']-shooting['x'][:,1:])
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(2*xnum,nSteps))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(2*xnum,nSteps))

            # classify the resulting constraints
            tmp_order_a = get_order(cs.vertcat(*model.ode),[syms[i] for i in range(pnum+xnum+ynum+anum+unum+vnum)]) # global order
            tmp_order_c = get_order(cs.vertcat(*model.ode),[syms[i] for i in range(pnum+xnum+ynum+anum+unum)]) # order considering the continuous variables only
            labels_tmp += 2*['<ocp_int'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(xnum)]


    # constraints for discrete transitions
    if model.itr != []:
        # define discrete transition
        itr = cs.Function('itr',[cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms,usyms,vsyms,dtsym)],[cs.vertcat(*model.itr)])

        # evaluate the discrete transitions
        itr = itr.map(nSteps)
        trans = itr(cs.vertcat(shooting['t'][:-1],
                               shooting['x'][:,:-1],
                               shooting['y'][:,:-1],
                               shooting['a'][:,:-1],
                               shooting['p'][:,:-1],
                               shooting['i'][:,:-1],
                               shooting['u'],
                               shooting['v'],
                               shooting['dt']))

        # add discrete transition constraints
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['y'][:,1:]-shooting['y'][:,:-1]-trans)
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.DM.zeros(trans.size()))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(trans.size()))


            # classify the resulting constraints
            tmp_order_a = get_order(cs.vertcat(*model.itr),[syms[i] for i in range(pnum+xnum+ynum+anum+unum+vnum)]) # global order
            tmp_order_c = get_order(cs.vertcat(*model.itr),[syms[i] for i in range(pnum+xnum+ynum+anum+unum)]) # order considering the continuous variables only
            labels_tmp += ['<ocp_dst'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(ynum)]

        else:

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],shooting['y'][:,1:]-shooting['y'][:,:-1]-trans,-shooting['y'][:,1:]+shooting['y'][:,:-1]+trans)
            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(trans.shape[0]*2,trans.shape[1]))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(trans.shape[0]*2,trans.shape[1]))


             # classify the resulting constraints
            tmp_order_a = get_order(cs.vertcat(*model.itr),[syms[i] for i in range(pnum+xnum+ynum+anum+unum+vnum)]) # global order
            tmp_order_c = get_order(cs.vertcat(*model.itr),[syms[i] for i in range(pnum+xnum+ynum+anum+unum)]) # order considering the continuous variables only
            labels_tmp += 2*['<ocp_dst'+str(i)+'>' + '<semi_cvx>'*int(tmp_order_a[i]>1)*int(tmp_order_c[i]<=1) + '<non-convex>'*int(tmp_order_c[i]>1) for i in range(ynum)]


    # reshape the constraints vector and add labels
    problem.cns['exp'] = cs.reshape(problem.cns['exp'],(-1,1))
    problem.cns['lob'] = cs.reshape(problem.cns['lob'],(-1,1))
    problem.cns['upb'] = cs.reshape(problem.cns['upb'],(-1,1))
    problem.cns['lbl'] = labels_tmp*nSteps



    # add initial constraints
    if model.icns != [] and opts['with_initial_cnss']:
        if not opts['leq0_only']:
            problem.cns['exp'] = cs.vertcat(cs.substitute(cs.vertcat(*[c.exp for c in model.icns]),\
                                                    cs.vertcat(psyms,xsyms,ysyms,asyms,isyms),\
                                                    cs.vertcat(shooting['p'][:,0],shooting['x'][:,0],shooting['y'][:,0],shooting['a'][:,0],shooting['i'][:,0])),\
                                      problem.cns['exp'])

            problem.cns['lob'] = cs.vertcat(cs.vertcat(*[c.lob for c in model.icns]),problem.cns['lob'])
            problem.cns['upb'] = cs.vertcat(cs.vertcat(*[c.upb for c in model.icns]),problem.cns['upb'])
            problem.cns['lbl'] = [model.icns[i].lbl+'<ocp_ic'+str(i)+'>' for i in range(len(model.icns))] + problem.cns['lbl']
        else:
            icnsexp = cs.vertcat(*([c.lob - c.exp for c in model.icns if c.lob > -cs.DM.inf()]+\
                                   [c.exp - c.upb for c in model.icns if c.upb <  cs.DM.inf()]))

            problem.cns['exp'] = cs.vertcat(cs.substitute(icnsexp,\
                                                    cs.vertcat(psyms,xsyms,ysyms,asyms,isyms),\
                                                    cs.vertcat(shooting['p'][:,0],shooting['x'][:,0],shooting['y'][:,0],shooting['a'][:,0],shooting['i'][:,0])),\
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
                                                    cs.vertcat(psyms,xsyms,ysyms,asyms,isyms),\
                                                    cs.vertcat(shooting['p'][:,-1],shooting['x'][:,-1],shooting['y'][:,-1],shooting['a'][:,-1],shooting['i'][:,-1])))


            problem.cns['lob'] = cs.vertcat(problem.cns['lob'],cs.vertcat(*[c.lob for c in model.fcns]))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.vertcat(*[c.upb for c in model.fcns]))
            problem.cns['lbl'] = problem.cns['lbl'] + [model.fcns[i].lbl+'<ocp_fc'+str(i)+'>' for i in range(len(model.fcns))]

        else:
            fcnsexp = cs.vertcat(*([c.lob - c.exp for c in model.fcns if c.lob > -cs.DM.inf()]+\
                                   [c.exp - c.upb for c in model.fcns if c.upb <  cs.DM.inf()]))

            problem.cns['exp'] = cs.vertcat(problem.cns['exp'],\
                                      cs.substitute(fcnsexp,\
                                                    cs.vertcat(psyms,xsyms,ysyms,asyms,isyms),\
                                                    cs.vertcat(shooting['p'][:,-1],shooting['x'][:,-1],shooting['y'][:,-1],shooting['a'][:,-1],shooting['i'][:,-1])))

            problem.cns['lob'] = cs.vertcat(problem.cns['lob'], -cs.DM.inf(fcnsexp.shape[0],1))
            problem.cns['upb'] = cs.vertcat(problem.cns['upb'],cs.DM.zeros(fcnsexp.shape[0],1))
            problem.cns['lbl'] = problem.cns['lbl'] +\
                            [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.fcns) if c.lob > -cs.DM.inf()] +\
                            [c.lbl + '<ocp_ic'+str(i)+'>' for i,c in enumerate(model.fcns) if c.upb <  cs.DM.inf()]



    ## Construct the objective
    if unpack_objective:
        problem.obj = {'exp':cs.MX.zeros(shooting['t'].numel()),'val':None}
    else:
        problem.obj = {'exp':cs.MX(0.0),'val':None}

    # add lagrangian objective
    if not model.lag is None:

        # define integrator
        l0 = cs.MX.sym('l0')
        intg = integrate_ode(l0,cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms,usyms,vsyms),[model.lag],dtsym,opts['integration_opts']).map(nSteps)

        # evaluate integration
        tmp = intg.call({'x0':cs.DM.zeros(nSteps).T,\
                         'p':cs.vertcat(shooting['t'][:-1],shooting['x'][:,:-1],shooting['y'][:,:-1],shooting['a'][:,:-1],shooting['p'][:,:-1],shooting['i'][:,:-1],shooting['u'],shooting['v'],shooting['dt'])})['xf']

        if unpack_objective:
            problem.obj['exp'][:-1] += tmp.T
        else:
            problem.obj['exp'] += cs.sum2(tmp)

    # add discrete penalties
    if not model.dpn is None:
        dpnf = cs.Function('dpn',[cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms,usyms,vsyms,dtsym)],[model.dpn]).map(nSteps)
        tmp = dpnf(cs.vertcat(shooting['t'][:-1],shooting['x'][:,:-1],shooting['y'][:,:-1],shooting['a'][:,:-1],shooting['p'][:,:-1],shooting['i'][:,:-1],shooting['u'],shooting['v'],shooting['dt']))

        if unpack_objective:
            problem.obj['exp'][:-1] += tmp.T
        else:
            problem.obj['exp'] += cs.sum2(tmp)

    # add mayer objective
    if not model.may is None:

        tmp = cs.vertcat(problem.obj['exp'],cs.substitute(model.may,\
                                                          cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms),\
                                                          cs.vertcat(shooting['t'][:,-1],shooting['p'][:,-1],shooting['x'][:,-1],shooting['y'][:,-1],shooting['a'][:,-1],shooting['i'][:,-1])))
        tmp = cs.substitute(model.may,\
                            cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms),\
                            cs.vertcat(shooting['t'][:,-1],shooting['p'][:,-1],shooting['x'][:,-1],shooting['y'][:,-1],shooting['a'][:,-1],shooting['i'][:,-1]))

        if unpack_objective:
            problem.obj['exp'][-1] += tmp
        else:
            problem.obj['exp'] += tmp








    # add sos1 constraints
    if not model.sos1 is None:
        problem.sos1 = []
        varsname = [v['nme'] for v in vardata]
        for c in range(len(model.sos1)):
            indxs = [i for i in range(len(varsname)) if varsname[i] in model.sos1[c].group]
            if model.sos1[c].weights != []:
                problem.sos1.append({'g': [[t*numVars + i for i in indxs] for t in range(nSteps)],\
                                     'w': [cs.substitute(model.sos1[c].weights,cs.vertcat(tsym,psyms,xsyms,ysyms,asyms,isyms),cs.vertcat(shooting['t'][:,-1],shooting['p'][:,-1],shooting['x'][:,-1],shooting['y'][:,-1],shooting['a'][:,-1],shooting['i'][:,-1])) for t in range(nSteps)]})
            else:
                problem.sos1.append({'g': [[t*numVars + i for i in indxs] for t in range(nSteps)],'w':[None]*nSteps})




    return
