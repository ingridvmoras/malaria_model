

### Imports ###
import numpy as np  # handle arrays
import pandas as pd
from scipy import integrate  # numerical integration
import joblib as jl

import seaborn as sns  # for plots

import plots
from scipy.stats import ttest_ind


def odeFunLimitImmunity(t, y, **kwargs):
    try:
        print("Executing odeFunLimitImmunity...")
        # Unpack variables passed through kwargs
        a, g, d, e, r, n_strains, sigma,tau = \
            kwargs['a'], kwargs['g'], kwargs['d'], kwargs['e'], kwargs['r'], kwargs['n_strains'], kwargs['sigma'], kwargs['tau']

        # Unpack state variables
        P = y[0:n_strains]  # P contains the first n_strains elements of y
        S = y[n_strains:-1]  # Slicing from n_strains to the second last element
        C = y[-1]  # Last element is C
       
        t = round(t*10)
 
        if t<round(kwargs['inf_times'][0]+tau*10):
            
           sigma_C= 1e-12 
        else:
           sigma_C= 1

       
        # Get the corresponding sigma value
        sigma_t = sigma[t]
        
        # ODEs
        dP = (r - S - C) * P
        dS = sigma_t * (a * P) - e * S
        dC=  (g * P.sum())- d * C
 
        
        dy = np.concatenate([dP, dS, [dC]])
        return dy
    except Exception as e:
        print(f"Error in odeFunLimitImmunity: {e}")
        raise


def initCond(params):
    try:
        print("Initializing conditions...")
        strains = np.zeros(params['n_strains'])
        strain_immunities = np.zeros(params['n_strains'])
        cross_immunity = [params['cross_immunity_start']]

        y0 = np.concatenate([
            strains,
            strain_immunities,
            cross_immunity,
        ])

        y0[0] = y0[0] + params['inoc']
        return y0
    except Exception as e:
        print(f"Error in initCond: {e}")
        raise

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
    min_state_var = 2e-5

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
    

    if y_out.t[-1] - y_out.t[0] >= p['year_duration'] - p['t_dry']:
       
        if y_out.t[0] < p['year_duration']:#first year 
            t_last_wet = p['t_dry'] - 1
            t_last_dry = p['year_duration'] - 1
        else:
            t_last_wet = np.floor(y_out.t[0] / p['year_duration']) * p['year_duration'] + p['t_dry'] - 1
            t_last_dry = np.floor(y_out.t[0] / p['year_duration']) * p['year_duration'] + p['year_duration'] - 1

        t_last_wet_i = np.argmin(np.abs(y_out.t - t_last_wet))
        t_last_dry_i = np.argmin(np.abs(y_out.t - t_last_dry))
        
        if y_out.y[0:p['n_strains'], -1].max() > min_state_var:
           
            #y_out.persister_t.append(y_out.t[t_last_dry_i])
            #y_out.persister_y.append(y_out.y[:, t_last_dry_i])

            y_out.persister_t.append(y_out.t[t_last_wet_i])
            y_out.persister_y.append(y_out.y[:, t_last_wet_i])
        else:
          
            y_out.non_persister_t.append(y_out.t[t_last_dry_i])
            y_out.non_persister_y.append(y_out.y[:, t_last_dry_i])

            y_out.non_persister_t.append(y_out.t[t_last_wet_i])
            y_out.non_persister_y.append(y_out.y[:, t_last_wet_i])

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

        if not inf_evt == p['n_strains']-2 and len(y_next.t) > 0 and y_next.t[-1]-y_next.t[0] >= p['year_duration'] - p['t_dry']:
            if y_next.t[0] < p['year_duration']:
                t_last_wet = p['t_dry'] - 1
                t_last_dry = p['year_duration'] - 1
            else:
                t_last_wet = np.floor(y_next.t[0] / p['year_duration']) * p['year_duration'] + p['t_dry'] - 1
                t_last_dry = np.floor(y_next.t[0] / p['year_duration']) * p['year_duration'] + p['year_duration'] - 1

            t_last_wet_i = np.argmin(np.abs(y_next.t - t_last_wet))
            t_last_dry_i = np.argmin(np.abs(y_next.t - t_last_dry))

            if y_next.y[0:p['n_strains'], -1].max() > min_state_var:
                #y_out.persister_t.append(y_next.t[t_last_dry_i])
                #y_out.persister_y.append(y_next.y[:, t_last_dry_i])

                y_out.persister_t.append(y_next.t[t_last_wet_i])
                y_out.persister_y.append(y_next.y[:, t_last_wet_i])
            else:
                #y_out.non_persister_t.append(y_next.t[t_last_dry_i])
                #y_out.non_persister_y.append(y_next.y[:, t_last_dry_i])

                y_out.non_persister_t.append(y_next.t[t_last_wet_i])
                y_out.non_persister_y.append(y_next.y[:, t_last_wet_i])
                
                
        y_out.t = np.concatenate([y_out.t,y_next.t])
        if len(y_next.y)>0:
            if y_next.y.ndim == 1:
                y_next.y = y_next.y.transpose()

            y_out.y = np.concatenate([y_out.y,y_next.y],axis=1)

    y_out.y = ( y_out.y > min_state_var/10 ) * y_out.y
    y_out.persister_y = ( np.array(y_out.persister_y) > min_state_var/10 ) * np.array(y_out.persister_y)
    y_out.non_persister_y = ( np.array(y_out.non_persister_y) > min_state_var/10 ) * np.array(y_out.non_persister_y)

    if persister_out:
        persister = pd.DataFrame(columns=['Time','Parasites','COI','Diversity','Evenness','Cross immunity','Persister'])
        if len(y_out.persister_y) > 0:
            persister['Time'] = y_out.persister_t
            persister['Parasites'] = y_out.persister_y[:,0:p['n_strains']].sum(axis=1)
            persister['COI'] = np.array( y_out.persister_y[:,0:p['n_strains']] > 0 ).sum(axis=1)
            frac = y_out.persister_y[:,0:p['n_strains']] / np.maximum( np.tile( persister['Parasites'], (p['n_strains'],1) ).transpose(),  1e-10 )
            persister['Diversity'] = - np.sum( frac * np.log( np.maximum( frac,1e-10 ) ), 1 )
            persister['Evenness'] = persister['Diversity'] / np.log( np.maximum( persister['COI'],2 ) )
            persister['Cross immunity'] = y_out.persister_y[:,-1]
            persister['Persister'] = True

        non_persister = pd.DataFrame(columns=['Time','Parasites','COI','Diversity','Evenness','Cross immunity','Persister'])
        if len(y_out.non_persister_y) > 0:
            non_persister['Time'] = y_out.non_persister_t
            non_persister['Parasites'] = y_out.non_persister_y[:,0:p['n_strains']].sum(axis=1)
            non_persister['COI'] = np.array( y_out.non_persister_y[:,0:p['n_strains']] > 0 ).sum(axis=1)
            frac = y_out.non_persister_y[:,0:p['n_strains']] / np.maximum( np.tile( non_persister['Parasites'], (p['n_strains'],1) ).transpose(),  1e-10)
            non_persister['Diversity'] = - np.sum( frac * np.log( np.maximum( frac,1e-10 ) ), 1 )
            non_persister['Evenness'] = non_persister['Diversity'] / np.log( np.maximum( non_persister['COI'],2 ) )
            non_persister['Cross immunity'] = y_out.non_persister_y[:,-1]
            non_persister['Persister'] = False


        return y_out, pd.concat([persister,non_persister],ignore_index=True)
    else:
        return y_out


def params(mean_infections_per_season=3, t_dry=181, years=20, year_duration=365):
    try:
        print("Setting up parameters...")
        params = {}
        strains_per_season = np.random.poisson(lam=mean_infections_per_season, size=years).astype(int)
        params['n_strains'] = strains_per_season.sum().astype(int) + 1
        params['t_0'] = 0
        params['t_f'] = year_duration * years
        params['t_den'] = 0.1
        params['t_dry'] = t_dry
        params['year_duration'] = year_duration
        params['inf_times'] = np.append(
            np.concatenate([np.sort(np.random.random(strains_per_season[y]) * t_dry + (y * year_duration)) for y in range(years)]),
            [year_duration * years]
        )
        params['sigma'] = 0
        params['sigma_C'] = 0
        params['inoc'] = 5.6e4 / 5e6
        params['cross_immunity_start'] = 0
        params['a'] = 7e-6
        params['g'] = 1e-6
        params['d'] = 3.7e-4
        params['r'] = np.log(14) / (2)
        params['K_s'] = 2
        params['K_c'] = 2
        params['odeFun'] = odeFunLimitImmunity
        params['tau'] = 14
        params['e'] = 1e-5

        print("Parameters set up successfully.")
        return params
    except Exception as e:
        print(f"Error in params: {e}")
        raise


def solveModel(persister_out=True, num_simulations=32):
    '''
    Main method containing multiple solver calls and concatenation of results.
    Writes figures to file.
    '''
    try:
        print("Starting simulations...")
        all_results = []
        all_dfs = []

        for sim in range(num_simulations):
            print(f"Running simulation {sim + 1}/{num_simulations}...")
            p = params()
            t = np.linspace(p['t_0'], p['t_f'], int((p['t_f'] - p['t_0']) / p['t_den']) + 1)

            # Initialize sigma as an array of zeros with the same length as t
            sigma = np.zeros_like(t)
            
            for i, inf_time in enumerate(p['inf_times'][:-1]):
                tau_end = inf_time + p['tau']
                next_inf_time = p['inf_times'][i + 1]

                sigma[(t > tau_end) & (t < next_inf_time)] = 1

            p['sigma'] = sigma

            y_0 = initCond(p)

            # Solve model
            sol, df = odeSolver(p['odeFun'], t, y_0, p, solver="RK45", persister_out=persister_out)

            # Assign simulation ID to the dataframe
            df['ID'] = sim

            all_results.append(sol)
            all_dfs.append(df)

        # Combine all simulation results into a single dataframe
        combined = pd.concat(all_dfs, ignore_index=True)

        # Aggregate results by simulation ID and persister status
        combined = combined.groupby(["ID", "Persister"]).agg('mean').reset_index()

        # Generate plots
        plots.figDistributions(combined, p, f_name='sigma_simulations')

        print("Simulations completed successfully.")
        return all_results, combined
    except Exception as e:
        print(f"Error in solveModel: {e}")
        raise


# Run the simulations
solveModel(num_simulations=120)


