#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 18:15:25 2018

@author: Massimo De Mauri
"""
import MIRT_OC as oc

def butcher_integrator(name,table,data,options=None):
    
    opts = {}
    opts['tf'] = 1
    opts['n_steps'] = 1
    opts['code_gen'] = False
        
    if not options is None:
        for k in options.keys():
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('butcher_integrator, option not recognized ' + k)
    
    assert(table.shape[0]==table.shape[1])
    table = oc.sparsify(table)
    

    
    A = table[:-1,1:]
    B = table[-1,1:]
    C = table[:-1,0]
    
    
    
    N = A.shape[0]
    nx = data['x'].numel()
    ns = opts['n_steps'] 

    
    x = oc.reshape(data['x'],-1,1)
    p = oc.reshape(data['p'],-1,1)

    Fode = oc.Function('ode',[x,p],[data['ode']])

    x0 = oc.MX.sym('x0',x.shape)
    xf = x0.T
    
    

    h = opts['tf']/ns


    # if the matrix is lower triangular: direct calculation
    if A.is_tril():
        
        K = oc.MX(N,nx)
        for s in range(ns):
            
            for i in range(N):
                
                K[i,:] = Fode(xf + A[i,:i]@K[:i,:]*h,p).T
                 
            xf += B@K*h
                                            
    else:
        
        Fode_map = Fode.map(N)
        v = oc.MX.sym('v',N*nx*ns,1)
        K = oc.reshape(v,N,nx*ns)
        cnss = oc.MX(nx*N*ns,1)
        
        xf = oc.horzcat(oc.repmat(x0.T,N,1),oc.MX(N,nx*ns))

        

        for s in range(ns):
            xf[:,(s+1)*nx:(s+2)*nx] = xf[:,s*nx:(s+1)*nx] + oc.repmat(B@K[:,s*nx:(s+1)*nx]*h,N,1)
            cnss[s*nx*N:(s+1)*nx*N] = oc.reshape(K[:,s*nx:(s+1)*nx] - Fode_map((xf[:,s*nx:(s+1)*nx] + A@K[:,s*nx:(s+1)*nx]*h).T,oc.repmat(p,1,N)).T,-1,1)
            


        # use a root finder
        f = oc.Function('f',[v,x0,p],[cnss,1])
        RF = oc.rootfinder('RF','newton',f)        
        
        v0 = oc.reshape(oc.repmat(Fode(x0,p).T,N,ns),-1,1)  # initial guess for the rootfinder
            
        vstar = RF(v0,x0,p)[0]
                          
        xf = oc.substitute(xf[-1,-nx:],v,vstar) 
            
        
    
    if opts['code_gen']:
        return 0
    else:
        return oc.Function(name,[x0,p],[xf.T],['x0','p'],['xf'])

    