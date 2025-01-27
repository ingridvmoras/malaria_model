import numpy as np
import pandas as pd
from ODE_Solver_persistence import params, initCond, odeSolver, figTSeries

def run_simulation(mean_infections_list, a_values, r_values, n_reps=10):
    results = []

    for mean_infections in mean_infections_list:
        for a in a_values:
            for r in r_values:
                # Set up model conditions
                p = params(mean_infections_per_season=mean_infections)
                p['a'] = a
                p['r'] = r
                y_0 = initCond(p)
                t = np.linspace(p['t_0'], p['t_f'], int((p['t_f'] - p['t_0']) / p['t_den']) + 1)

                # Solve model
                sol = odeSolver(p['odeFun'], t, y_0, p, solver="RK45")

                # Collect results
                results.append({
                    'mean_infections': mean_infections,
                    'a': a,
                    'r': r,
                    'solution': sol
                })

    return results

if __name__ == "__main__":
    mean_infections_list = np.linspace(1,15,15) # Example values
    a_values = np.linspace(5e-6,9e-6,30)  # Example values for 'a'
    r_values = [np.log(10)/2, np.log(14)/2]  # Example values for 'r'

    results = run_simulation(mean_infections_list, a_values, r_values)
    analyze_results(results)