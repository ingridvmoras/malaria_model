from ODE_Solver_persistence import initCond, odeSolver, odeFunNoLimitImmunity
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(19680801)
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
    params['r'] = np.log(14)/(2)
    params['K_s'] = 2
    params['K_c'] = 2
    params['odeFun'] = odeFunNoLimitImmunity
    
    return params

def sweep_mean_infections( mean_infections_list):
    results = []
    for i in tqdm(mean_infections_list):
        p = params(mean_infections_per_season=i)
        sol = solveModel(p)  
        results.append((i, sol))
    return results

def sweep_parameters(params, parameter, values):
    results = []
    for value in tqdm(values):
        params = params.copy()
        params[parameter] = value
        sol = solveModel(params)  
        results.append((value, sol))
    return results

def runs(param,params,times, persister_out=False):
    results = []
    distributions = []
    for i in tqdm(range(times)):
        p = params.copy()
        p['r'] = np.log(np.random.uniform(1, 14, p['n_strains']))/2
        sol, df = solveModel(p, persister_out=persister_out)
        results.append((i+1, df))
        for r in p['r']:
            distributions.append((i+1, r))
    ds = pd.DataFrame(distributions, columns=['simulation', 'r'])
    return results, ds
    
def runs_a(params, times, distribution='uniform', sd=[1],persister_out=False):
    results= []
   
    
    if distribution=='normal':
    
        distributions=[] 
        for i in tqdm(range(times)): 
            p = params.copy()
            p['a'] = np.random.normal(7 * (10**-6), sd[0], p['n_strains'])
            sol,df = solveModel(p,persister_out=persister_out)
            results.append((i+1, df))
            for a_value in p['a']:
                distributions.append((i+1, a_value))
        ds = pd.DataFrame(distributions, columns=['simulation', 'a'])
        return results, ds
    
    elif distribution=='multiNorm':
        distributions=[] 
        for i in tqdm(range(len(sd))): 
            p = params.copy()
            p['a'] = np.random.normal(7 * (10**-6), sd[i], p['n_strains'])
            sol,df = solveModel(p,persister_out=persister_out) 
            results.append((sd[i], df))
            for a_value in p['a']:
                distributions.append((i+1, a_value))
        ds = pd.DataFrame(distributions, columns=['simulation', 'a'])
        return results, ds
    
    else: 
        
        distributions=[]
        for i in tqdm(range(times)):
            p = params.copy()
            p['a'] = np.random.uniform(7 * (10**-7), 7 * (10**-5), p['n_strains'])
            sol = solveModel(p,persister_out=persister_out) 
            results.append((i+1, sol))
            for a_value in p['a']:
                distributions.append((i+1, a_value))
        return results,df
    
def sweep_inf_alpha(param1, values1, values2):
    results = []
    for value1 in tqdm(values1):
        p = params(mean_infections_per_season=value1)
        for value2 in values2:
            p = p.copy()
            p[param1] = value2
            sol = solveModel(p)
            results.append((value1, value2, sol))
    return results

def mixed_results(results):
    data = []
    for value1, value2, sol in results:
        parasites, cross_im, coi, diversity, evenness = shannon_diversity(sol, sol.y.shape[0] - 1)
        for i in range(len(parasites)):
            data.append({
                'mean_infections': value1,
                'a': value2,
                'parasites': parasites[i],
                'coi': coi[i],
                'diversity': diversity[i],
                'evenness': evenness[i],
                'cross_im': cross_im[i]
            })
    df = pd.DataFrame(data)
    return df

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

def solveModel(p=None,persister_out=False):

    '''
    Main method containing single solver and plotter calls.
    Writes figures to file.
    '''

    # Set up model conditions
    if not params:
        p = params() # get parameter values, store in dictionary p
    
    y_0 = initCond(p) # get initial conditions
    t = np.linspace(p['t_0'],p['t_f'],int((p['t_f']-p['t_0'])/p['t_den']) + 1)
        # time vector based on minimum, maximum, and time step values

    # Solve model
 
    #parasites,cross_im,coi,diversity,evenness =shannon_diversity(sol,p['n_strains'])

    #plots.figTSeries(sol, coi, diversity,evenness,p,f_name='ODE_tseries_persistence_single8.png')


    # print(np.array(sol.persister_t).astype('int'))
    # # print(sol.persister_y)

    # # Call plotting of figure 1
    
    return odeSolver(p['odeFun'],t,y_0,p,solver="RK45",persister_out=persister_out)





