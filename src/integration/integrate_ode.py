#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:19:40 2017

@author: Massimo De Mauri
"""

import MIRT_OC as oc

def integrate_ode(x,p,ode,dt,options = None):

    opts = {}
    opts['schema'] = 'feuler'
    opts['n_steps'] = 1
    
    
        
    if not options is None:
        for k in options:
            if k in opts:
                opts[k] = options[k]
            else:
                oc.warn('Option not recognized : ' + k )
        
             
    if opts['schema'] == 'cvodes':
                          
        intg = oc.integrator('F','cvodes',{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1})
        
    elif opts['schema'] == 'rk':
        
        intg = oc.integrator('F','rk',{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'number_of_finite_elements':opts['n_steps']})
        
    elif opts['schema'] == 'e_euler':
        
        table = oc.DM([[0,0],[0,1]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']})
        
    elif opts['schema'] == 'i_euler':
        
        table = oc.DM([[1,1],[0,1]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']})  
        
    elif opts['schema'] == 'rk4':
        
        table = oc.DM([[0,0,0,0,0],[.5,.5,0,0,0],[.5,0,.5,0,0],[1,0,0,1,0],[0,1/6,1/3,1/3,1/6]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']})    
    
    elif opts['schema'] == 'ralston':  
        
        table = oc.DM([[0,0,0],[2/3,2/3,0],[0,.25,.75]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']}) 

    elif opts['schema'] == 'gl2':
        
        table = oc.DM([[.5,.5],[0,1]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']})
        
    elif opts['schema'] == 'gl4':

        table = oc.DM([[.5+oc.sqrt(3)/6,.25,.25+oc.sqrt(3)/6],[.5-oc.sqrt(3)/6,.25-oc.sqrt(3)/6,.25],[0,.5,.5]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']}) 
    
    elif opts['schema'] == 'gl6':
        
        table = oc.DM([[.5-oc.sqrt(15)/10,5/36,2/9-oc.sqrt(15)/15,5/36-oc.sqrt(15)/30],[.5,5/36+oc.sqrt(15)/24,2/9,5/36-oc.sqrt(15)/24],[.5+oc.sqrt(15)/10,5/36+oc.sqrt(15)/30,2/9+oc.sqrt(15)/15,5/36],[0,5/18,4/9,5/18]])
        intg = oc.butcher_integrator('F',table,{'x':x,'p':oc.vertcat(p,dt),'ode':dt*oc.vertcat(*ode)},{'tf':1,'n_steps':opts['n_steps']})    
        
        
    elif type(opts['schema']) == type(oc.DM()):

        oc.butcher_integrator(table,x,p,ode,dt,{'n_steps':opts['n_steps']})
        
    else:
        raise NameError(opts['schema'] + ' not implemented')
        return None


    return intg