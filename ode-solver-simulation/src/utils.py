from ODE_Solver_persistence import solveModel

def sweep_mean_infections(params, mean_infections_list):
    results = []
    for mean_infections in mean_infections_list:
        params = params.copy()
        params['mean_infections_per_season'] = mean_infections
        sol = solveModel(params)  
        results.append((mean_infections, sol))
    return results

def sweep_parameters(params, parameter, values):
    results = []
    for value in values:
        params = params.copy()
        params[parameter] = value
        sol = solveModel(params)  
        results.append((value, sol))
    return results

def runs(params,times):
    results= []
    for i in range(times):
        sol= solveModel(params)
        results.append((i, sol))
    return results



def plot_results(results):
    # Placeholder for plotting logic
    # This function can be expanded to visualize the results of the simulations
    pass