#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Jun 12 16:10:42 2017

@author: Massimo De Mauri
'''

import csv
import casadi as cs
import numpy as np
import MIRT_OC as oc

import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import matplotlib as mpl
from matplotlib import cm
from scipy.interpolate import interp1d
from os import path, chdir
from time import time

# local components lib
from models import insight_50kw_power_jc as engine
from models import advisor_em_pwr_sc as electric_motor
from models import battery_hu as battery


current_directory = path.abspath(path.dirname(__file__))
chdir(current_directory)

# ------------------------------------------------
# Cycle
# ----------------------------------------------

# import the driving cycle
with open(current_directory+'/cycle.csv', 'r') as csvfile:
    cycle_dat = list(csv.reader(csvfile,delimiter = ','))[0]
    cycle_dat = [float(s) for s in cycle_dat]
    cycle_dat = cycle_dat[10:]


# resolution of the time discretization
dt = 1
# start time
start = 7
# length of the considered time window
wl = len(cycle_dat)-start-1 # int(min(200,len(cycle_dat)-start-1))

cycle_base  = cs.DM(cycle_dat[start:wl+start+1])
t_base = np.linspace(0,wl,wl+1)
f = interp1d(np.squeeze(t_base),np.squeeze(cycle_base),axis=0)


n_Steps = wl/dt

time_vec = t_base[0] + cs.DM([t for t in range(int(n_Steps+1))])*(t_base[-1]-t_base[0])/n_Steps
cycle_dat = cs.DM(f(time_vec))
dcycle_dat = cs.vertcat((cycle_dat[1:]-cycle_dat[:-1])/(time_vec[1:]-time_vec[:-1]),0)


# ------------------------------------------------
# Model
# ------------------------------------------------
base_mass = 800
frictionCoeff = 0.005
wheelR = 0.285
drag_area = 2
air_density = 1.225
drag_coeff = .35

# temporary variables
w = cs.MX.sym('w')
P = cs.MX.sym('P')
Prat = cs.MX.sym('Prat')
Erat = cs.MX.sym('Erat')
SoC = cs.MX.sym('Erat')


# ICE
ICE_wrat = 6000*np.pi/30
ICE_data = engine()
ICE_dFin = cs.Function('ICE_dFin',        [Prat,w,P], [ICE_data['dFin'](ICE_wrat,1000*Prat,w,1000*P)])
ICE_Fstart = cs.Function('ICE_Fstart',  [Prat],     [ICE_data['Fstart'](1000*Prat)])
ICE_Pmax = cs.Function('ICE_Pmax',      [Prat,w],   [ICE_data['Pmax'](ICE_wrat,1000*Prat,w)*1e-3])
ICE_mass = cs.Function('ICE_mass',      [Prat],     [ICE_data['mass'](1000*Prat)])
ICE_minw = max(ICE_data['minw'](ICE_wrat),1000*np.pi/30)

# EM
EM_wrat = 10000*np.pi/30
EM_data = electric_motor()
EM_Pin = cs.Function('EM_Pin',  [Prat,w,P], [EM_data['Pin'](EM_wrat,1000*Prat,w,1000*P)*1e-3])
EM_Pmax = cs.Function('EM_Pmax',[Prat,w],   [EM_data['Pmax'](EM_wrat,1000*Prat,w)*1e-3])
EM_mass = cs.Function('EM_mass',[Prat],     [EM_data['mass'](1000*Prat)])




# battery
BT_data = battery()
SoC_max = BT_data['SoC_max']
SoC_min = BT_data['SoC_min']
BT_dSoC = cs.Function('BT_dSoC',[Erat,SoC,P],   [BT_data['dSoC'](Erat*3.6e6,SoC,1000*P)])
BT_Pout = cs.Function('BT_Pout',[Erat,SoC,P],   [BT_data['Pout'](Erat*3.6e6,SoC,1000*P)*1e-3])
BT_Pmax = cs.Function('BT_Pmax',[Erat,SoC],     [BT_data['Pmax'](Erat*3.6e6,SoC)*1e-3])
BT_Pmin = cs.Function('BT_Pmin',[Erat,SoC],     [BT_data['Pmin'](Erat*3.6e6,SoC)*1e-3])
BT_mass = cs.Function('BT_mass',[Erat],         [BT_data['mass'](Erat*3.6e6)])



# ------------------------------------------------
# Parameters initial values
# ------------------------------------------------

# speeds to guess ratios
referece_speeds = [15,30,55,85,115]

# for diesel:
referece_ICEspeed_diesel = 2000
R_diesel = [(referece_ICEspeed_diesel*np.pi/30)/((referece_speeds[k]/3.6)/wheelR) for k in range(len(referece_speeds))]

# for gasoline
referece_ICEspeed_gasoline = 2500
R_gasoline = [(referece_ICEspeed_gasoline*np.pi/30)/((referece_speeds[k]/3.6)/wheelR) for k in range(len(referece_speeds))]

# from advisor default parallel
Rem_advisor = 0.99*(EM_wrat/ICE_wrat)
R_advisor = [13.33,7.57,5.01,3.77,2]



# ------------------------------------------------
# Problem definition
# ------------------------------------------------

# fixed parameters
R = R_gasoline
Rem = (3000*np.pi/30)/((60/3.6)/wheelR)
ICE_Prat = 55
EM_Prat = 25
BT_Erat = 1.024 #kWh
TNK_Frat = 30000
TNK_Finit = TNK_Frat/2
SoC_opt  =  SoC_min + 0.75*(SoC_max - SoC_min)
SoC_start = SoC_min + 0.1*(SoC_max - SoC_min)




# create optimal control model
model = oc.oc_problem('Parallel Hybrid Drivetrain')

# external inputs
model.i = [oc.input('cycle',cycle_dat), oc.input('dcycle',dcycle_dat)]
cycle = model.i[0].sym
dcycle = model.i[1].sym
SoCcoeff = oc.input('SoCcoeff',oc.DM.zeros(time_vec.numel()))
model.i.append(SoCcoeff)

# differential states
SoC = oc.variable('SoC',SoC_min,SoC_max,SoC_start)
Fuel = oc.variable('F',0,TNK_Frat,TNK_Finit)
model.x = [SoC,Fuel]

# discrete transition states
OFFstate = oc.variable('OFFstate',0,1,1)
GEARstate = oc.variable('GEARstate',0,5,0)
model.y =  [OFFstate,GEARstate]

# algebraic variables
model.a = []

# continuous controls
dFuel = oc.variable('dF',0,20.0,0)
Pb = oc.variable('Pb',BT_Pmin(BT_Erat,1.0),BT_Pmax(BT_Erat,1.0),0)
Pice = oc.variable('Pice',0,ICE_Prat,0)
Pem = oc.variable('Pem',-EM_Prat,EM_Prat,0)
switch = oc.variable('switch',0,oc.inf,0)
model.u = [dFuel,Pice,Pb,Pem,switch]

#discrete controls
OFF = oc.variable('OFF',0,1,1)
G = [oc.variable('G'+str(k+1),0,1,int(k==0)) for k in range(len(R))]
model.v = [OFF] + G

# ode
model.ode = [BT_dSoC(BT_Erat,SoC.sym,Pb.sym),-dFuel.sym]

# discrete transitions
gear = sum([(k+1)*G[k].sym for k in range(len(R))])
model.itr = [OFF.sym-OFFstate.sym, gear-GEARstate.sym]

# precalculations
mass = base_mass + ICE_mass(ICE_Prat) + EM_mass(EM_Prat) + BT_mass(BT_Erat)
Freq = mass*dcycle+.5*air_density*drag_coeff*drag_area*cycle**2
Treq = Freq*wheelR
Paux = 0.3 # accessory power load kW
Preq = Freq*cycle/1000


wem = Rem*cycle/wheelR
wice = sum([G[k].sym*R[k]*cycle/wheelR for k in range(len(R))])
GEAR_ = sum([(k+1)*G[k].sym for k in range(len(R))])

# path constraints
model.pcns =  [oc.geq(Pem.sym+Pice.sym,Preq)]+\
              \
              [oc.eq(sum([G[k].sym for k in range(len(R))]) + OFF.sym,1)]+\
              [oc.leq(OFF.sym-OFFstate.sym, switch.sym)]+\
              [oc.geq(OFF.sym-OFFstate.sym,-switch.sym)]+\
              [oc.leq(GEAR_-GEARstate.sym, 5*switch.sym)]+\
              [oc.geq(GEARstate.sym-GEAR_,-5*switch.sym)]+\
              \
              [oc.leq(wice,ICE_wrat*0.5)]+\
              [oc.leq(ICE_minw*(1-OFF.sym),wice)]+\
              [oc.leq(Pice.sym,ICE_Prat*(1-OFF.sym))]+\
              [oc.leq(Pice.sym,ICE_Pmax(ICE_Prat,wice)[k]) for k in range(ICE_Pmax(0,0).numel())]+\
              [oc.leq(ICE_dFin(ICE_Prat,wice,Pice.sym)[k]-OFF.sym*ICE_dFin(ICE_Prat,0,0)[k],dFuel.sym) for k in range(ICE_dFin(0,0,0).numel())]+\
              \
              [oc.leq(Pb.sym,BT_Pmax(BT_Erat,SoC.sym))]+\
              [oc.geq(Pb.sym,BT_Pmin(BT_Erat,SoC.sym))]+\
              [oc.leq(EM_Pin(EM_Prat,wem,Pem.sym)[k] + Paux,BT_Pout(BT_Erat,SoC.sym,Pb.sym)) for k in range(EM_Pin(0,0,0).numel())]+\
              \
              [oc.leq(Pem.sym, EM_Prat)]+\
              [oc.geq(Pem.sym,-EM_Prat)]

# sos1 constraints
# model.sos1 = [oc.sos1_constraint([v.nme for v in G]+['OFF'],cs.vertcat(*[r.sym for r in R]+[0]))]

# objective
model.lag = (0.4*45.6*dFuel.sym+Pb.sym)
model.dpn = 0.4*45.6*ICE_Fstart(ICE_Prat)*switch.sym
model.may = 200*(-cs.log((SoC.sym-SoC_min)/(SoC_max-SoC_min)) - cs.log((SoC_max-SoC.sym)/(SoC_max-SoC_min)) + 2*cs.log(.5))

# model.may = (200*BT_Erat*SoCcoeff.sym*(1-SoC.sym))**2 # 200g/kWh is the maximum theoretical efficiency for engines
model.epigraph_reformulation(max_order = 2,integration_opts={'schema':'rk4','n_steps':1})

print(model)
print('\n---------------------------------------------------------------------------------')


# define expression for SoCcoeff
SoCcoeff_f = cs.Function('SoCcoeff',[v.sym for v in model.x+model.y],[1.0 - (SoC_max-SoC_min)/(SoC_max-SoC.sym + SoC_max-SoC_min)],
                                    [v.nme for v in model.x+model.y],['out'])


# solve
modes = [0]
save_graphs = False
integration_opts = {'schema':'rk4','n_steps':1}
stats = {}


RTL = 0 # relaxed tail length
PHL = 15 # prediction window length
shift_size = 1
num_iterations = 12
subsolverName = 'CLP'
conservativism_level = 2
nlpProcesses = 1
mipProcesses = 1
shift_style = None
shift_style = "relaxationOnly"
shift_style = "warmStart"
# shift_style = "fullRTI"

# create container for results
results = {'t':cs.DM(list(range(num_iterations-1+PHL+1)))};
for i in model.i:
    results[i.nme] = cs.DM.zeros(num_iterations-1+PHL+1)
for v in model.x + model.y + model.a:
    results[v.nme] = cs.DM.zeros(num_iterations-1+PHL+1)
for v in model.u+model.v:
    results[v.nme] = cs.DM.zeros(num_iterations-1+PHL)


# solve
for mode in modes:
    if mode == 0: # new MI-MPC using MIRT-OC

        fig_name = 'MIRTOC_'+str(start)+':'+str(dt)+':'+str(start+wl)
        mpc_options = {'max_iteration_time':oc.inf,
                       'OpenBB_opts':{'verbose':True,'conservativismLevel':conservativism_level,"relativeGapTolerance":1e-4,
                                      'nlpProcesses':nlpProcesses,'mipProcesses':mipProcesses,'nlpStepType':('OAnlpStep',),
                                      'nlpSettings':{'subsolverName':'IPOPT','constr_viol_tol':1e-5},
                                      'mipSettings':{'verbose':True,
                                                     'withBoundsPropagation':False,
                                                     'expansionPriorityRule':("lower_dualTradeoff",),
                                                     'subsolverSettings':{'subsolverName':subsolverName}}},
                       'integration_opts':integration_opts,
                       'prediction_horizon_length':PHL,'relaxed_tail_length':RTL,
                       'printLevel':1}

        # generate the mpc controller
        mpc_machine = oc.MPCmachine(model,mpc_options)

        # define the first measured state
        measured_state = {'SoC':SoC_start,'F':TNK_Finit,'OFFstate':1.0,'GEARstate':0.0}

        # collect the first parameters values
        input_values = {'t':time_vec[:PHL+RTL+1],
                        'cycle':cycle_dat[:PHL+RTL+1],
                        'dcycle':dcycle_dat[:PHL+RTL+1],
                        'SoCcoeff':SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(PHL+RTL+1)}


        # MPC iterations
        k0 = 0
        start_time = time()
        for i in range(num_iterations):

            print('MPC: Iteration =',i+1)

            # perform mpc iteration
            iteration_objective, iteration_results = mpc_machine.iterate(measured_state,input_values,shift_style,0)

            print("MPC: Optimal Objective =",iteration_objective)

            # fill in the newest results
            for v in model.x + model.y + model.a:
                results[v.nme][k0:k0+PHL+1] = iteration_results[v.nme][:PHL+1]
            for v in model.u + model.v:
                results[v.nme][k0:k0+PHL] = iteration_results[v.nme][:PHL]

            # update the initial timestep
            k0 += shift_size

            # update the measured state
            for v in model.x: measured_state[v.nme] = results[v.nme][k0]
            for v in model.y: measured_state[v.nme] = round(float(results[v.nme][k0]))

            # update the input values
            input_values = {'t':time_vec[k0:k0+PHL+RTL+1],
                            'cycle':cycle_dat[k0:k0+PHL+RTL+1],
                            'dcycle':dcycle_dat[k0:k0+PHL+RTL+1],
                            'SoCcoeff':cs.vertcat(input_values['SoCcoeff'][shift_size:],
                                                  SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(shift_size))
                            }


    elif mode == 1: # standard MI-MPC using bonmin

        fig_name = 'bonmin_'+str(start)+':'+str(dt)+':'+str(start+wl)
        mpc_options = {'max_iteration_time':oc.inf,
                       'minlp_solver_opts':{'printLevel':0,'primalTolerance':1e-4,'mi_solver_name':'Cplex'},
                       'integration_opts':integration_opts,
                       'prediction_horizon_length':PHL,'relaxed_tail_length':RTL,
                       'printLevel':1}


        # generate the mpc controller
        mpc_machine = oc.MPCmachine_bonmin(model,mpc_options)

        # MPC iterations
        k0 = 0

        # define the first measured state
        measured_state = {'SoC':SoC_start,'F':TNK_Finit,'OFFstate':1.0,'GEARstate':0.0}

        # collect the first parameters values
        input_values = {'t':time_vec[:PHL+RTL+1],
                        'cycle':cycle_dat[:PHL+RTL+1],
                        'dcycle':dcycle_dat[:PHL+RTL+1],
                        'SoCcoeff':SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(PHL+RTL+1)}

        start_time = time()
        for i in range(num_iterations):

            print('Iteration: ',i+1)

            # perform mpc iteration
            iteration_results = mpc_machine.iterate(measured_state,input_values)

            # fill in the newest results
            for v in model.x + model.y + model.a:
                results[v.nme][k0:k0+PHL+1] = iteration_results[v.nme][:PHL+1]
            for v in model.u + model.v:
                results[v.nme][k0:k0+PHL] = iteration_results[v.nme][:PHL]
            for v in model.i:
                results[v.nme][k0:k0+PHL+1] = input_values[v.nme][:PHL+1]

            # update the initial timestep
            k0 += shift_size

            # update the measured state
            for v in model.x + model.y:
                measured_state[v.nme] = results[v.nme][k0]

            # update the input values
            input_values = {'t':time_vec[k0:k0+PHL+RTL+1],
                            'cycle':cycle_dat[k0:k0+PHL+RTL+1],
                            'dcycle':dcycle_dat[k0:k0+PHL+RTL+1],
                            'SoCcoeff':cs.vertcat(input_values['SoCcoeff'][shift_size:],
                                                  SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(shift_size))
                           }



    elif mode == 2: # standard MI-MPC using POA

        fig_name = 'POA_'+str(start)+':'+str(dt)+':'+str(start+wl)
        mpc_options = {'max_iteration_time':oc.inf,
                       'minlp_solver_opts':{'printLevel':0,'primalTolerance':1e-4,'mi_solver_name':'Cplex'},
                       'integration_opts':integration_opts,
                       'prediction_horizon_length':PHL,'relaxed_tail_length':RTL,
                       'printLevel':1}


        # generate the mpc controller
        mpc_machine = oc.MPCmachine_POA(model,mpc_options)

        # MPC iterations
        k0 = 0

        # define the first measured state
        measured_state = {'SoC':SoC_start,'F':TNK_Finit,'OFFstate':1.0,'GEARstate':0.0}

        # collect the first parameters values
        input_values = {'t':time_vec[:PHL+RTL+1],
                        'cycle':cycle_dat[:PHL+RTL+1],
                        'dcycle':dcycle_dat[:PHL+RTL+1],
                        'SoCcoeff':SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(PHL+RTL+1)}

        start_time = time()
        for i in range(num_iterations):

            print('Iteration: ',i+1)

            # perform mpc iteration
            iteration_results = mpc_machine.iterate(measured_state,input_values)

            # fill in the newest results
            for v in model.x + model.y + model.a:
                results[v.nme][k0:k0+PHL+1] = iteration_results[v.nme][:PHL+1]
            for v in model.u + model.v:
                results[v.nme][k0:k0+PHL] = iteration_results[v.nme][:PHL]
            for v in model.i:
                results[v.nme][k0:k0+PHL+1] = input_values[v.nme][:PHL+1]

            # update the initial timestep
            k0 += shift_size

            # update the measured state
            for v in model.x + model.y:
                measured_state[v.nme] = results[v.nme][k0]

            # update the input values
            input_values = {'t':time_vec[k0:k0+PHL+RTL+1],
                            'cycle':cycle_dat[k0:k0+PHL+RTL+1],
                            'dcycle':dcycle_dat[k0:k0+PHL+RTL+1],
                            'SoCcoeff':cs.vertcat(input_values['SoCcoeff'][shift_size:],
                                                  SoCcoeff_f.call(measured_state)['out']*cs.DM.ones(shift_size))
                           }




    for k in range(len(R)):
        results['R'+str(k+1)] = R[k]
    results['Rem'] = Rem



    print('\n---------------------------------------------------------------------------------')
    print('Solution time: ',time()-start_time)
    print('Timings: ',mpc_machine.stats['times'])
    print('#Solves: ',mpc_machine.stats['num_solves'])
    print('---------------------------------------------------------------------------------\n')



    time_vec = results['t']
    Treq_f = cs.Function('Treq',[cs.vertcat(*[i.sym for i in model.i])],[Treq])
    Treq_f = Treq_f.map(time_vec.numel())
    Treq_v = Treq_f(oc.list_horzcat([results[i.sym.name()] for i in model.i]).T).T


    plt.close(fig_name)
    fig = plt.figure(fig_name)


    base_font ={'family' : 'sans','size'   : 12}
    mpl.rc('font', **base_font)
    font_titles = {'rotation':'vertical','fontsize':18,'va':'center','ha':'center','weight':'bold'}
    font_units = {'rotation':'vertical','fontsize':14,'va':'center','ha':'center'}
    font_results = {'weight':'bold','size':20,'ha':'center','va':'center'}


    text_space = 0.015
    x0 = 0.15
    xf = .85
    y0 = 0.09
    yf = 0.99

    n_plots = 5

    rem = results['Rem']
    rice = sum([results['R'+str(i+1)]*results['G'+str(i+1)] for i in range(len(R))])
    wr = cycle_dat[:time_vec.numel()]/wheelR
    dsc_labels = ['OFF','G1','G2','G3','G4','G5']



    # base layout
    ax_ = fig.add_axes([x0+2*text_space,y0,(xf-x0)-6*text_space,yf-y0],frameon = True)
    ax_.xaxis.set_visible(True)
    ax_.yaxis.set_visible(False)
    ax_.set_ylim([0,1])
    ax_.grid(axis='x')
    ax_.autoscale(enable=True,axis='x',tight=True)
    ax_.patch.set_alpha(0)


    # prepare the space
    units = []
    titles = []
    ax = []
    for p in range(n_plots):
        ax.append(fig.add_axes([x0+2*text_space,y0 + p*(yf-y0)/n_plots,(xf-x0)-6*text_space,(yf-y0)/n_plots],sharex=ax_,frameon = False))
        ax[-1].yaxis.set_visible(True)
        ax[-1].xaxis.set_visible(False)
        ax[-1].set_axisbelow(True)
        ax[-1].patch.set_alpha(0)
        ax[-1].yaxis.tick_right()
        ax[-1].yaxis.set_label_position('right')
        ax[-1].grid(axis='y')
        ax[-1].autoscale(enable=True,axis='x',tight=True)
        titles.append('')
        units.append('')




    # powers
    data2 = results['Pem']
    data3 = results['Pice']
    data1 = data2 + data3

    min_data = oc.dm_min(cs.vertcat(data1,data2,data3))
    max_data = oc.dm_max(cs.vertcat(data1,data2,data3))
    ticks_period = 10

    yticks = [ticks_period*t for t in range(int(oc.dm_round(min_data/ticks_period+.1)),int(oc.dm_round(max_data/ticks_period-.1))+1)]
    ax[4].set_yticks(yticks)
    ax[4].set_ylim(min(yticks)-.6*ticks_period,max(yticks)+.6*ticks_period)
    titles[4] = 'Powers'
    units[4] = 'kW'

    ax[4].step(time_vec,cs.vertcat(data1[0],data1), label = 'Requested', color='k',linewidth=2)
    ax[4].step(time_vec,cs.vertcat(data2[0],data2), label='EM',linewidth=2,color=(0,0,0.7))
    ax[4].step(time_vec,cs.vertcat(data3[0],data3), label='ICE (after GB)',linewidth=2,color=(0.9,0,0))

    ax[4].legend(loc='upper left',ncol = 3,fontsize = 14)


    # speeds
    data1 = wr[:-1]*rem*30/np.pi
    data2 = wr[:-1]*rice*30/np.pi
    data3 = ICE_wrat*cs.DM.ones(wr.numel()-1,1)*30/np.pi


    min_data = oc.dm_min(cs.vertcat(data1,data2))
    max_data = oc.dm_max(cs.vertcat(data1,data2))

    if max_data > 2000:
        ticks_period = 1000
    else:
        ticks_period = 500


    yticks = [ticks_period*t for t in range(int(oc.dm_round(min_data/ticks_period+.1)),int(oc.dm_round(max_data/ticks_period-.1))+1)]
    ax[3].set_yticks(yticks)
    ax[3].set_ylim(min(yticks)-.6*ticks_period,max(yticks)+.6*ticks_period)
    titles[3] = 'Speeds'
    units[3] = 'RpM'

    ax[3].step(time_vec,cs.vertcat(data1[0],data1),  label='EM',linewidth=2,color=(0,0,0.7))
    ax[3].step(time_vec,cs.vertcat(data2[0],data2), label='ICE (before GB)',linewidth=2,color=(0.9,0,0))
    ax[3].legend(loc='upper left',ncol = 3,fontsize = 14)



    # SoC
    data = 100*results['SoC']
    min_data = oc.dm_min(data);
    max_data = oc.dm_max(data)
    ticks_period = int(2*(max_data-min_data)+.5)/10

    yticks = [ticks_period*t for t in range(int(oc.dm_round(min_data/ticks_period+.1)),int(oc.dm_round(max_data/ticks_period-.1))+1)]
    ax[2].set_yticks(yticks)
    ax[2].set_ylim(min(yticks)-.6*ticks_period,max(yticks)+.6*ticks_period)
    titles[2] = 'SoC'
    units[2] = '%'

    ax[2].plot(oc.squeeze(time_vec),oc.squeeze(data),linewidth=2,color=(.4,.6,0))


    # discrete variables
    dsc_ass = cs.DM()
    for i,l in enumerate(dsc_labels):
        dsc_ass = oc.horzcat(dsc_ass,results[l])



    ax[1].set_ylim(-.5,len(dsc_labels)+.5)
    yticks = [0] + [t+1 for t in range(len(dsc_labels))]
    ax[1].set_yticks(yticks)
    ax[1].set_ylim(min(yticks)-.5,max(yticks)+.5)
    ax[1].set_yticklabels('')
    ax[1].set_yticks([t +.5 for t in range(len(dsc_labels))],minor=True)
    ax[1].set_yticklabels(dsc_labels,minor=True)

    titles[1] = 'Discrete C.'


    cmap = cm.nipy_spectral
    colors  = [(.2,.2,.2)] +[cmap(int(i*cmap.N/len(dsc_labels))) for i in range(1,len(dsc_labels))]

    for i in range(len(dsc_labels)):
        for t in range(time_vec.numel()-1):
            if dsc_ass[t,i] > 1e-4:
                ax[1].add_patch(ptc.Rectangle(xy=(time_vec[t],i),width=time_vec[t+1]-time_vec[t],height=dsc_ass[t,i],facecolor=colors[i],edgecolor='k',fill=True))
                ax_.add_patch(ptc.Rectangle(xy=(time_vec[t],0),width=time_vec[t+1]-time_vec[t],height=1,facecolor=list(colors[i])[:3]+ [float(.1*dsc_ass[t,i])],edgecolor='none',fill=True))


    # driving cycle
    data = 3.6*cycle_dat[:time_vec.numel()]
    min_data = oc.dm_min(data)
    max_data = oc.dm_max(data)
    ticks_period = 10

    yticks = [ticks_period*t for t in range(int(oc.dm_round(min_data/ticks_period+.1)),int(oc.dm_round(max_data/ticks_period-.1))+1)]
    ax[0].set_yticks(yticks)
    ax[0].set_ylim(min(yticks)-.6*ticks_period,max(yticks)+.6*ticks_period)
    titles[0] = 'Cycle'
    units[0] = 'km/h'

    ax[0].plot(time_vec,data,linewidth=2,color='k')





    ax0 = fig.add_axes([0,0,1,1],frameon=False)
    for p in range(n_plots):
        rect = ptc.Rectangle(xy=(x0,y0+p*(yf-y0)/n_plots),width=xf-x0,height=(yf-y0)/n_plots,facecolor='w',edgecolor='k',fill=False)
        ax0.add_patch(rect)
        ax0.text(x0+text_space,y0+(p+.5)*(yf-y0)/n_plots,titles[p],font_titles)
        ax0.text(xf-text_space,y0+(p+.5)*(yf-y0)/n_plots,units[p],font_units)

    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    if save_graphs:
        plt.savefig(current_directory+'/tmp_'+oc.path.basename(__file__)[:-3]+'_results/'+ '_' + fig_name+'.png', bbox_inches='tight')
