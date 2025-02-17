#!/usr/bin/env python3

'''
Deterministic numerical solver for ODE systems
Pablo Cardenas R.

used for
Zhang et al., 2023
Coded by github.com/pablocarderam based on original model by An-Ni Zhang
'''


### Imports ###
import numpy as np # handle arrays
import pandas as pd
from scipy import integrate # numerical integration
import joblib as jl
import itertools as it

import seaborn as sns # for plots
import matplotlib.pyplot as plt
import matplotlib.colors as mlc





### Methods ###
# User-defined methods #
def odeFunLimitImmunity(t,y,**kwargs):

    """
    Contains system of differential equations.

    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolanting functions, etc.
    Returns:
        Dictionary containing dY/dt for the given state and parameter values
    """

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    a,g,d,r,K_s,K_c,n_strains = \
        kwargs['a'],kwargs['g'],kwargs['d'],kwargs['r'],kwargs['K_s'],kwargs['K_c'],kwargs['n_strains']

    # P,S,C = y # unpack state variables
    P = y[0:n_strains]  # P contains the first n_strains elements of y
    S = y[n_strains:-1]  # Slicing from n_strains to the second last element
    C = y[-1]
    # (state variable order matters for numerical solver)

    # ODEs
    dP = ( r - S - C * P / P.sum() ) * P
    # dS = a * ( 1 - S.sum() / K_s ) * P
    # dC = g * ( 1 - C / K_c ) * P - d * C
    dS = a * P * P / P.sum()
    dC = g * P.sum() - d * C

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = np.concatenate([dP,dS,[dC]])

    return dy


def odeFunNoLimitImmunity(t,y,**kwargs):

    """
    Contains system of differential equations.

    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolanting functions, etc.
    Returns:
        Dictionary containing dY/dt for the given state and parameter values
    """

    # Unpack variables passed through kwargs (see I thought I could avoid this
    # and I just made it messier)
    a,g,d,r,K_s,K_c,n_strains = \
        kwargs['a'],kwargs['g'],kwargs['d'],kwargs['r'],kwargs['K_s'],kwargs['K_c'],kwargs['n_strains']

    # P,S,C = y # unpack state variables
    P = y[0:n_strains]
    S = y[n_strains:-1]
    C = y[-1]
    # (state variable order matters for numerical solver)

    # ODEs
    dP = ( r - S - C ) * P
    # dS = a * ( 1 - S.sum() / K_s ) * P
    # dC = g * ( 1 - C / K_c ) * P - d * C
    dS = a * P
    dC = g * P.sum() - d * C

    # Gather differential values in list (state variable order matters for
    # numerical solver)
    dy = np.concatenate([dP,dS,[dC]])

    return dy



#params = {
    #     't_0':0,          # h - Initial time value
    #     't_f': 0,      # h - Final time value
    #     't_den':0.1,      # h - Size of time step to evaluate with

    #     't_dry':0,
    #     'year_duration':0,

    #     'n_strains': 0,
    #     'inoc':5.6e4 / 5e6,
    #     'inf_times': 0,
    #     'cross_immunity_start':0,

    #     # 'a':7e-6*np.ones(n_strains),       # 1/generation - infected bacteria death rate
    #     # 'g':1e-6*np.ones(n_strains),     # 1/(PFU/mL * h) - phage-bacteria infection rate
    #     'a':7e-6,       # 1/generation - infected bacteria death rate
    #     'g':1e-6,     # 1/(PFU/mL * h) - phage-bacteria infection rate
    #     'd':3.7e-4,     # 1/(PFU/mL * h) - phage-bacteria infection rate
    #     'r':np.log(14)/(2),  # CFU/mL - max bacteria growth rate
    #     'K_s':2,          # 1/generation - spacer loss rate
    #     'K_c':2,       # 1/(CFU/mL * h) - HGT rate of spacer

    #     'odeFun':odeFunNoLimitImmunity,
    # }


def params(mean_infections_per_season=3, t_dry=181, years=20, year_duration=365):

    """
    Returns default values constant values for the model in a dictionary.
    """
    params={}
    strains_per_season = np.random.poisson( lam=mean_infections_per_season, size=years ).astype(int)
    params['n_strains'] = strains_per_season.sum()
    params['t_0'] = 0
    params['t_f'] = year_duration*years
    params['t_den'] = 0.1
    params['t_dry'] = t_dry
    params['year_duration'] = year_duration
    params['inf_times'] = np.append( np.concatenate([ np.sort( np.random.random(strains_per_season[y]) * t_dry + ( y * year_duration ) ) for y in range(years) ]), [year_duration*years] )
    params['inoc'] = 5.6e4 / 5e6
    params['cross_immunity_start'] = 0
    params['a'] = 7e-6
    params['g'] = 1e-6
    params['d'] = 3.7e-4
    params['r'] = np.log(14) 
    params['K_s'] = 2
    params['K_c'] = 2
    params['odeFun'] = odeFunNoLimitImmunity
    
    return params


def initCond(params):

    '''
    Return initial conditions values for the model in a dictionary.
    '''

    strains=np.zeros(params['n_strains'])
    strain_immunities=np.zeros(params['n_strains'])
    cross_immunity=[params['cross_immunity_start']]

    y0 = np.concatenate([
        # Initial concentrations in [M] (order of state variables matters)
        strains,  # CFU/mL - susceptible, uninfected bacteria S
        strain_immunities,              # CFU/mL - infected bacteria I
        cross_immunity,              # CFU/mL - Resistant bacteria with spacer R
        ])

    y0[0] = y0[0] + params['inoc']

    return y0

def shannon_diversity(sol,n_strains):
    parasites = np.array(sol.y[0:n_strains, :].sum(axis=0))
    cross_im= sol.y[-1, :]
    P = sol.y[0: n_strains, :]
    coi = np.array(sol.y[0:n_strains, :] > 0).sum(axis=0)
    sum_P = P.sum(axis=0)
    sum_P[sum_P < 1] = 1
    frac = P / sum_P
    shannon_diversity = -np.sum(frac * np.log(np.maximum(frac, 1e-10)), axis=0)
    shannon_evenness = shannon_diversity / np.log(np.maximum(coi, 2))
    return parasites,cross_im,coi,shannon_diversity,shannon_evenness



# Pre-defined methods #
# These shouldn't have to be modified for different models
def odeSolver(func,t,y0,p,solver='LSODA',rtol=1e-8,atol=1e-8,persister_out=False,**kwargs):

    """
    Numerically solves ODE system.

    Arguments:
        func     : function with system ODEs
        t        : array with time span over which to solve
        y0       : array with initial state variables
        p        : dictionary with system constant values
        solver   : algorithm used for numerical integration of system ('LSODA'
                   is a good default, use 'Radau' for very stiff problems)
        rtol     : relative tolerance of solver (1e-8)
        atol     : absolute tolerance of solver (1e-8)
        **kwargs : additional parameters to be used by ODE function (i.e.,
                   interpolation)
    Outputs:
        y : array with state value variables for every element in t
    """

    # default settings for the solver
    options = { 'RelTol':10.**-8,'AbsTol':10.**-8 }
    min_state_var = 1e-2

    # takes any keyword arguments, and updates options
    options.update(kwargs)

    # runs scipy's new ode solver
    y_out = integrate.solve_ivp(
            lambda t_var,y: func(t_var,y,**p,**kwargs), # use a lambda function
                # to sub in all parameters using the ** double indexing operator
                # for dictionaries, pass any additional arguments through
                # **kwargs as well
            [t[0],p['inf_times'][0]], # initial and final time values
            y0, # initial conditions
            method=solver, # solver method
            t_eval=t[ (t >= t[0]) * (t < p['inf_times'][0]) ], # time point vector at which to evaluate
            # rtol=rtol, # relative tolerance value
            # atol=atol # absolute tolerance value
        )

    y_out.persister_t = []
    y_out.persister_y = []
    y_out.non_persister_t = []
    y_out.non_persister_y = []

    if y_out.t[-1]-y_out.t[0] >= p['year_duration'] - p['t_dry']:
        t_last_dry = np.floor( y_out.t[0] / p['year_duration'] ) * p['year_duration'] + p['t_dry']
        t_last_dry_start_i = np.argmin( y_out.t < t_last_dry )
        if y_out.y[0:p['n_strains'],-1].max() > min_state_var:
            y_out.persister_t.append(y_out.t[t_last_dry_start_i])
            y_out.persister_y.append(y_out.y[:,t_last_dry_start_i])
        else:
            y_out.non_persister_t.append(y_out.t[t_last_dry_start_i])
            y_out.non_persister_y.append(y_out.y[:,t_last_dry_start_i])

    for inf_evt in range(p['n_strains']-1):
        new_y0 = np.maximum( y_out.y[:,-1],0 ).flatten()
        new_y0[inf_evt+1] = new_y0[inf_evt+1] + p['inoc']
        # runs scipy's new ode solver
        y_next = integrate.solve_ivp(
                lambda t_var,y: func(t_var,y,**p,**kwargs), # use a lambda function
                    # to sub in all parameters using the ** double indexing operator
                    # for dictionaries, pass any additional arguments through
                    # **kwargs as well
                [p['inf_times'][inf_evt],p['inf_times'][inf_evt+1]], # initial and final time values
                new_y0, # initial conditions
                method=solver, # solver method
                t_eval=t[ (t >= p['inf_times'][inf_evt]) * (t < p['inf_times'][inf_evt+1]) ], # time point vector at which to evaluate
                # rtol=rtol, # relative tolerance value
                # atol=atol # absolute tolerance value
            )

        # print(y_next.y[0:p['n_strains'],-1].max() > min_state_var)
        # print( y_next.t[-1]-y_next.t[0] >= p['year_duration'] - p['t_dry'])
        # print( inf_evt == p['n_strains']-2)
        if not inf_evt == p['n_strains']-2 and len(y_next.t) > 0 and y_next.t[-1]-y_next.t[0] >= p['year_duration'] - p['t_dry']:
            t_last_dry = np.floor( y_next.t[0] / p['year_duration'] ) * p['year_duration'] + p['t_dry']
            t_last_dry_start_i = np.argmin( y_next.t < t_last_dry )
            if y_next.y[0:p['n_strains'],-1].max() > min_state_var:
                y_out.persister_t.append(y_next.t[t_last_dry_start_i])
                y_out.persister_y.append(y_next.y[:,t_last_dry_start_i])
            else:
                y_out.non_persister_t.append(y_next.t[t_last_dry_start_i])
                y_out.non_persister_y.append(y_next.y[:,t_last_dry_start_i])

        y_out.t = np.concatenate([y_out.t,y_next.t])
        if len(y_next.y)>0:
            if y_next.y.ndim == 1:
                print(y_next.y.shape)
                y_next.y = y_next.y.transpose()

            y_out.y = np.concatenate([y_out.y,y_next.y],axis=1)


    y_out.y = ( y_out.y > min_state_var/10 ) * y_out.y
    y_out.persister_y = ( np.array(y_out.persister_y) > min_state_var/10 ) * np.array(y_out.persister_y)
    # print(y_out.persister_y.shape)
    y_out.non_persister_y = ( np.array(y_out.non_persister_y) > min_state_var/10 ) * np.array(y_out.non_persister_y)
    # print(y_out.non_persister_y.shape)
    # np.savetxt("y_out_persister_y.csv", y_out.persister_y, delimiter=",")
    # np.savetxt("y_out.y.csv", y_out.y, delimiter=",")

    if persister_out:
        persister = pd.DataFrame(columns=['Time','Parasites','COI','Diversity','Evenness','Cross immunity','Persister'])
        if len(y_out.persister_y) > 0:
            persister['Time'] = y_out.persister_t
            persister['Parasites'] = y_out.persister_y[:,0:p['n_strains']].sum(axis=1)
            persister['COI'] = np.array( y_out.persister_y[:,0:p['n_strains']] > 0 ).sum(axis=1)
            frac = y_out.persister_y[:,0:p['n_strains']] / np.maximum( np.tile( persister['Parasites'], (p['n_strains'],1) ).transpose(), 1 )
            persister['Diversity'] = - np.sum( frac * np.log( np.maximum( frac,1e-10 ) ), 1 )
            persister['Evenness'] = persister['Diversity'] / np.log( np.maximum( persister['COI'],2 ) )
            persister['Cross immunity'] = y_out.persister_y[:,-1]
            persister['Persister'] = True

        non_persister = pd.DataFrame(columns=['Time','Parasites','COI','Diversity','Evenness','Cross immunity','Persister'])
        if len(y_out.non_persister_y) > 0:
            non_persister['Time'] = y_out.non_persister_t
            non_persister['Parasites'] = y_out.non_persister_y[:,0:p['n_strains']].sum(axis=1)
            non_persister['COI'] = np.array( y_out.non_persister_y[:,0:p['n_strains']] > 0 ).sum(axis=1)
            frac = y_out.non_persister_y[:,0:p['n_strains']] / np.maximum( np.tile( non_persister['Parasites'], (p['n_strains'],1) ).transpose(), 1 )
            non_persister['Diversity'] = - np.sum( frac * np.log( np.maximum( frac,1e-10 ) ), 1 )
            non_persister['Evenness'] = non_persister['Diversity'] / np.log( np.maximum( non_persister['COI'],2 ) )
            non_persister['Cross immunity'] = y_out.non_persister_y[:,-1]
            non_persister['Persister'] = False

        return pd.concat([persister,non_persister],ignore_index=True)
    else:
        return y_out
    



# Solving model
# To generate data, uncomment the following...
# Single timecourse
def solveModel(params):

    '''
    Main method containing single solver and plotter calls.
    Writes figures to file.
    '''

    # Set up model conditions
    if not params:
        p = params() # get parameter values, store in dictionary p
    else:
        p = params
    y_0 = initCond(p) # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
    sol = odeSolver(p['odeFun'],t,y_0,p,solver="RK45")
   

    # print(np.array(sol.persister_t).astype('int'))
    # # print(sol.persister_y)

    # # Call plotting of figure 1
    
    return sol

   #figTSeries(sol,p,f_name='ODE_tseries_persistence_single6.png')

    
    #plt.close()
    

#sol= solveModel()

# multiSim

# def runReplicate():
#     p = params() # get parameter values, store in dictionary p
#     y_0 = initCond(p) # get initial conditions
#     t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
#         # time vector based on minimum, maximum, and time step values
#
#     return odeSolver(p['odeFun'],t,y_0,p,solver="RK45",persister_out=True)
#
# def multiSim(n_reps=32):
#
#     '''
#     Main method containing single solver and plotter calls.
#     Writes figures to file.
#     '''
#
#     # Set up model conditions
#     p = params() # get parameter values, store in dictionary p
#     y_0 = initCond(p) # get initial conditions
#     t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
#         # time vector based on minimum, maximum, and time step values
#
#     # Solve model
#     n_cores = jl.cpu_count()
#
#     sols = jl.Parallel(n_jobs=n_cores, verbose=10) (
#         jl.delayed( runReplicate ) () for _ in range(n_reps)
#          )
#
#     dat = pd.concat(sols,ignore_index=True)
#     dat.to_csv('persistence.csv')
#     print(dat)
#
#     # Call plotting of figure 1
#     figDistributions(sol,p,f_name='ODE_persistence.png')
#
#     plt.close()
#
# multiSim()


#
# # Heatmaps!
# # First heatmap is K and d_t
# samples = 200 # heatmap width in pixels
#
# # These are the values of parameters used in sweeps (have to fill in axis ticks manually)
# K_drops = np.power( 10, np.linspace( 6, 10, samples, endpoint=True ) )
# print(K_drops)
# print(np.power( 10, np.linspace( 6, 10, 5, endpoint=True ) ))
#
# dilution_times = 2 * np.power( 10, np.linspace( 1, -1, samples, endpoint=True ) )
# print(dilution_times)
# print(np.linspace( 2, 0, 5, endpoint=True ))
# print(2 * np.power( 10, np.linspace( 1, -1, 5, endpoint=True ) ))
# # These times are then shown as frequencies
#
# # stores parameters and values to be used in sweep
# param_sweep_dic = { 'd_t':dilution_times,'K':K_drops }
#
# # generate dataframe with all combinations
# params_list = param_sweep_dic.keys()
# value_lists = [ param_sweep_dic[param] for param in params_list ]
# combinations = list( it.product( *value_lists ) )
# param_df = pd.DataFrame(combinations)
# param_df.columns = params_list
#
# results = {} # store results
#
# # This runs a single pixel
# def run(param_values):
#     # Set up model conditions
#     p = params() # get parameter values, store in dictionary p
#     y_0 = initCond() # get initial conditions
#     t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
#         # time vector based on minimum, maximum, and time step values
#
#     for i,param_name in enumerate(params_list):
#         p[param_name] = param_values[i]
#
#     # Solve model
#     sol = odeSolver(odeFun,t,y_0,p,solver="LSODA");
#
#     res_frac = sol.y[2,:]/(sol.y[2,:]+sol.y[0,:])
#     t_10 = np.argmax(res_frac>0.1) * params()['t_den']
#
#     return [res_frac[-1],t_10]
#
# # Parallelize running all pixels in heatmap
# n_cores = jl.cpu_count()
#
# res = jl.Parallel(n_jobs=n_cores, verbose=10) (
#     jl.delayed( run ) (param_values) for param_values in combinations
#      )
#
# dat = param_df
#
# dat['res_frac'] = np.array(res)[:,0]
# dat['t_10'] = np.array(res)[:,1]
# dat.to_csv('crispr_heatmaps_Kd.csv')
# # ...until here
#
# dat = pd.read_csv('crispr_heatmaps_Kd.csv')
#
# # Reformat data for heatmaps
# dat['d_t'] = 1/dat['d_t']
# print('Rates: ')
# print(list(dat['d_t'].unique()))
# dat_frac_res = dat.pivot(index='K',columns='d_t',values='res_frac')
# dat_frac_t10 = dat.pivot(index='K',columns='d_t',values='t_10')
#
# # Plot heatmaps
# plt.rcParams.update({'font.size': 12})
#
# def plotHeatmap(
#         dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
#     plt.figure(figsize=(8,8), dpi=200)
#     ax = plt.subplot(1, 1, 1)
#     if cmap == 'magma':
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
#             )
#     else:
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
#             )
#     ax.figure.axes[-1].yaxis.label.set_size(15)
#     ax.invert_yaxis()
#     spacing = '\n\n\n'# if show_labels else ''
#     plt.ylabel('Maximum bacterial density (CFU/mL)'+spacing,fontsize=15)
#     plt.xlabel(spacing+'Dilution rate (1/generation)',fontsize=15)
#     plt.savefig(file_name, bbox_inches='tight')
#
# plotHeatmap(
#     dat_frac_res,'Fraction of bacterial population with spacer',
#     'crispr_heatmap_deterministic_Kd_frac.png','viridis_r', vmin=0, vmax=1#, show_labels=True
#     )
#
# plotHeatmap(
#     dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
#     'crispr_heatmap_deterministic_Kd_t10.png','magma',  vmin=1e-1, vmax=1e2,#, show_labels=True
#     )
#
#
# # Second heatmap is h and c
#
# samples = 200 # heatmap width in pixels
#
# h_drops = (params()['b']/0.24) * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )
#     # fitness costs tested in Opqua stochastic model
# print(h_drops)
# print((params()['b']/0.24) * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) ))
# # contact rates tested in Opqua stochastic model
# c_drops = (params()['c']/0.24) * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )*10
# # print( np.linspace( 3, 1, samples, endpoint=True ) )
# # print(np.power( 5, -np.linspace( 3, 1, samples, endpoint=True ) ))
# print(c_drops)
# print((params()['c']/0.24) * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) )*10)
#
# # stores parameters and values to be used in sweep
# param_sweep_dic = { 'c':c_drops,'h':h_drops }
#
# # generate dataframe with all combinations
# params_list = param_sweep_dic.keys()
# value_lists = [ param_sweep_dic[param] for param in params_list ]
# combinations = list( it.product( *value_lists ) )
# param_df = pd.DataFrame(combinations)
# param_df.columns = params_list
#
# results = {} # store results
#
# # Parallelize running all pixels in heatmap
# n_cores = jl.cpu_count()
#
# res = jl.Parallel(n_jobs=n_cores, verbose=10) (
#     jl.delayed( run ) (param_values) for param_values in combinations
#      )
#
# dat = param_df
#
# dat['res_frac'] = np.array(res)[:,0]
# dat['t_10'] = np.array(res)[:,1]
# dat.to_csv('crispr_heatmaps_HGT.csv')
# # ...until here
#
# dat = pd.read_csv('crispr_heatmaps_HGT.csv')
#
# # Reformat data for heatmaps
# dat_frac_res = dat.pivot(index='c',columns='h',values='res_frac')
# dat_frac_t10 = dat.pivot(index='c',columns='h',values='t_10')
#
# # Plot heatmaps
# plt.rcParams.update({'font.size': 12})
#
# def plotHeatmap(
#         dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
#     plt.figure(figsize=(8,8), dpi=200)
#     ax = plt.subplot(1, 1, 1)
#     if cmap == 'magma':
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
#             )
#     else:
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
#             )
#     ax.figure.axes[-1].yaxis.label.set_size(15)
#     ax.invert_yaxis()
#     spacing = '\n\n\n'# if show_labels else ''
#     plt.xlabel(spacing+'Spacer horizontal transfer rate (1/(CFU/mL * gen.))',fontsize=15)
#     plt.ylabel('Spacer direct acquisition rate (1/generation)'+spacing,fontsize=15)
#     plt.savefig(file_name, bbox_inches='tight')
#
# plotHeatmap(
#     dat_frac_res,'Fraction of bacterial population with spacer',
#     'crispr_heatmap_deterministic_HGT_frac.png','viridis_r'#, show_labels=True
#     )
#
# plotHeatmap(
#     dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
#     'crispr_heatmap_deterministic_HGT_t10.png','magma', vmin=1e-1, vmax=1e2,#, show_labels=True
#     )
#
# # Last one spacer loss
#
# samples = 200 # heatmap width in pixels
#
# K_drops = np.power( 10, np.linspace( 6, 10, samples, endpoint=True ) )
#     # fitness costs tested in Opqua stochastic model
# print(K_drops)
# print(np.power( 10, np.linspace( 6, 10, 5, endpoint=True ) ))
# # contact rates tested in Opqua stochastic model
# loss_rates = 1e-4 * np.power( 10, np.linspace( -2, 2, samples, endpoint=True ) )
# # print( np.linspace( 3, 1, samples, endpoint=True ) )
# # print(np.power( 5, -np.linspace( 3, 1, samples, endpoint=True ) ))
# print(loss_rates)
# print(1e-4 * np.power( 10, np.linspace( -2, 2, 5, endpoint=True ) ))
#
# # stores parameters and values to be used in sweep
# param_sweep_dic = { 'l':loss_rates,'K':K_drops }
#
# # generate dataframe with all combinations
# params_list = param_sweep_dic.keys()
# value_lists = [ param_sweep_dic[param] for param in params_list ]
# combinations = list( it.product( *value_lists ) )
# param_df = pd.DataFrame(combinations)
# param_df.columns = params_list
#
# results = {} # store results
#
# # # Parallelize running all pixels in heatmap
# # n_cores = jl.cpu_count()
# #
# # res = jl.Parallel(n_jobs=n_cores, verbose=10) (
# #     jl.delayed( run ) (param_values) for param_values in combinations
# #      )
# #
# # # for param_values in combinations:
# #     # run(param_values)
# #
# # dat = param_df
# #
# # dat['res_frac'] = np.array(res)[:,0]
# # dat['t_10'] = np.array(res)[:,1]
# # dat.to_csv('crispr_heatmaps_loss.csv')
# # ...until here
#
# dat = pd.read_csv('crispr_heatmaps_loss.csv')
#
# # Reformat data for heatmaps
# dat_frac_res = dat.pivot(index='K',columns='l',values='res_frac')
# dat_frac_t10 = dat.pivot(index='K',columns='l',values='t_10')
#
# # Plot heatmaps
# plt.rcParams.update({'font.size': 12})
#
# def plotHeatmap(
#         dat,cmap_lab,file_name,cmap, vmin=None, vmax=None, show_labels=False):
#     plt.figure(figsize=(8,8), dpi=200)
#     ax = plt.subplot(1, 1, 1)
#     if cmap == 'magma':
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, norm=mlc.LogNorm()
#             )
#     else:
#         ax = sns.heatmap(
#             dat, linewidth = 0 , annot = False, cmap=cmap,
#             cbar_kws={'label': cmap_lab}, vmin=vmin, vmax=vmax,
#             xticklabels=show_labels, yticklabels=show_labels, #norm=mlc.LogNorm()
#             )
#     ax.figure.axes[-1].yaxis.label.set_size(15)
#     ax.invert_yaxis()
#     spacing = '\n\n\n'# if show_labels else ''
#     plt.ylabel('Maximum bacterial density (CFU/mL)'+spacing,fontsize=15)
#     plt.xlabel(spacing+'Spacer loss rate (1/generation)',fontsize=15)
#     plt.savefig(file_name, bbox_inches='tight')
#
# plotHeatmap(
#     dat_frac_res,'Fraction of bacterial population with spacer',
#     'crispr_heatmap_deterministic_loss_frac.png','viridis_r'#, show_labels=True
#     )
#
# plotHeatmap(
#     dat_frac_t10,'Time at which resistant bacteria with spacer reach 10% (generations)',
#     'crispr_heatmap_deterministic_loss_t10.png','magma'#, show_labels=True
#     )
