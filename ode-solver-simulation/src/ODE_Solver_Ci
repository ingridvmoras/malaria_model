### Imports ###
import numpy as np  # handle arrays
import pandas as pd
from scipy import integrate  # numerical integration
import joblib as jl
import itertools as it

import seaborn as sns  # for plots
import matplotlib.pyplot as plt
import matplotlib.colors as mlc

import plots
import utils
from scipy.stats import ttest_ind


def odeFunLimitImmunity(t, y, **kwargs):
    """
    Contains system of differential equations.
    Arguments:
        t        : current time variable value
        y        : current state variable values (order matters)
        **kwargs : constant parameter values, interpolating functions, etc.
    Returns:
        Array containing dY/dt for the given state and parameter values
    """

    # Unpack variables passed through kwargs
    a, g, d, e, r, n_strains = \
        kwargs['a'], kwargs['g'], kwargs['d'], kwargs['e'], kwargs['r'], kwargs['n_strains']

    # Unpack state variables
    P = y[0:n_strains]  # P contains the first n_strains elements of y
    S = y[n_strains:-1]  # Slicing from n_strains to the second last element
    C = y[-1]  # Last element is C

    t = round(t * 10)

    # ODEs
    dP = (r - S - C) * P
    dS = (a * P) - e * S
    dC = g * P.sum() - d * C

    # Concatenate derivatives into a single array
    dy = np.concatenate([dP, dS, [dC]])
    return dy


def initCond(params):
    '''
    Return initial conditions values for the model in a dictionary.
    '''

    strains = np.zeros(params['n_strains'])
    strain_immunities = np.zeros(params['n_strains'])
    cross_immunity = [params['cross_immunity_start']]

    y0 = np.concatenate([
        # Initial concentrations in [M] (order of state variables matters)
        strains,  # CFU/mL - susceptible, uninfected bacteria S
        strain_immunities,  # CFU/mL - infected bacteria I
        cross_immunity,  # CFU/mL - Resistant bacteria with spacer R
    ])

    y0[0] = y0[0] + params['inoc']

    return y0


def Hill(K, n, L):
    """
    Hill function to calculate the probability of successful infection.
    Arguments:
        K : float - Half-saturation constant (controls the steepness of the curve).
        n : float - Hill coefficient (controls the sensitivity to changes in L).
        L : float - Cross immunity value (input to the function).
    Returns:
        float - Probability of successful infection.
    """
    epsilon = 1e-10  # Small value to avoid division by zero
    L_safe = np.maximum(L, epsilon)
    return 1 - (1 / (1 + (K / L_safe)**n))


def odeSolver(func, t, y0, p, solver='LSODA', rtol=1e-8, atol=1e-8, persister_out=False, **kwargs):
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

    # Default settings for the solver
    options = {'RelTol': 10.**-8, 'AbsTol': 10.**-8}
    min_state_var = 2e-5

    # Takes any keyword arguments, and updates options
    options.update(kwargs)

    # Runs scipy's new ODE solver
    y_out = integrate.solve_ivp(
        lambda t_var, y: func(t_var, y, **p, **kwargs),  # Use a lambda function
        [t[0], p['inf_times'][0]],  # Initial and final time values
        y0,  # Initial conditions
        method=solver,  # Solver method
        t_eval=t[(t >= t[0]) * (t < p['inf_times'][0])],  # Time points to evaluate
    )

    y_out.persister_t = []
    y_out.persister_y = []
    y_out.non_persister_t = []
    y_out.non_persister_y = []

    if y_out.t[-1] - y_out.t[0] >= p['year_duration'] - p['t_dry']:
        if y_out.t[0] < p['year_duration']:  # First year
            t_last_wet = p['t_dry'] - 1
            t_last_dry = p['year_duration'] - 1
        else:
            t_last_wet = np.floor(y_out.t[0] / p['year_duration']) * p['year_duration'] + p['t_dry'] - 1
            t_last_dry = np.floor(y_out.t[0] / p['year_duration']) * p['year_duration'] + p['year_duration'] - 1

        t_last_wet_i = np.argmin(np.abs(y_out.t - t_last_wet))
        t_last_dry_i = np.argmin(np.abs(y_out.t - t_last_dry))

        if y_out.y[0:p['n_strains'], -1].max() > min_state_var:
            y_out.persister_t.append(y_out.t[t_last_wet_i])
            y_out.persister_y.append(y_out.y[:, t_last_wet_i])
        else:
            y_out.non_persister_t.append(y_out.t[t_last_dry_i])
            y_out.non_persister_y.append(y_out.y[:, t_last_dry_i])

            y_out.non_persister_t.append(y_out.t[t_last_wet_i])
            y_out.non_persister_y.append(y_out.y[:, t_last_wet_i])

    for inf_evt in range(p['n_strains'] - 1):
        # Calculate current cross immunity (last value of cross immunity)
        cross_immunity = y_out.y[-1, -1]  # Last value of cross immunity

        # Calculate the probability of successful infection using the Hill function
        infection_probability = Hill(K=p['K_hill'], n=p['n_hill'], L=cross_immunity)

        # Generate a random value between 0 and 1
        random_value = np.random.random()

        # Check if a new infection event occurs
        if random_value > infection_probability:
            # Update initial conditions for the new infection event
            new_y0 = np.maximum(y_out.y[:, -1], 0).flatten()
            new_y0[inf_evt + 1] = new_y0[inf_evt + 1] + p['inoc']

            # Solve the ODE system for the new event
            y_next = integrate.solve_ivp(
                lambda t_var, y: func(t_var, y, **p, **kwargs),
                [p['inf_times'][inf_evt], p['inf_times'][inf_evt + 1]],
                new_y0,
                method=solver,
                t_eval=t[(t >= p['inf_times'][inf_evt]) * (t < p['inf_times'][inf_evt + 1])],
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
                    

            # Update results if a valid solution is generated
            if len(y_next.t) > 0:
                y_out.t = np.concatenate([y_out.t, y_next.t])
                if y_next.y.ndim == 1:
                    y_next.y = y_next.y.transpose()
                y_out.y = np.concatenate([y_out.y, y_next.y], axis=1)
        else:
            # If no infection event occurs, maintain previous values
            y_out.t = np.concatenate([y_out.t, [p['inf_times'][inf_evt + 1]]])
            y_out.y = np.concatenate([y_out.y, y_out.y[:, -1].reshape(-1, 1)], axis=1)

    y_out.y = (y_out.y > min_state_var / 10) * y_out.y
    y_out.persister_y = (np.array(y_out.persister_y) > min_state_var / 10) * np.array(y_out.persister_y)
    y_out.non_persister_y = (np.array(y_out.non_persister_y) > min_state_var / 10) * np.array(y_out.non_persister_y)

    if persister_out:
        persister = pd.DataFrame(columns=['Time', 'Parasites', 'COI', 'Diversity', 'Evenness', 'Cross immunity', 'Persister'])
        if len(y_out.persister_y) > 0:
            persister['Time'] = y_out.persister_t
            persister['Parasites'] = y_out.persister_y[:, 0:p['n_strains']].sum(axis=1)
            persister['COI'] = np.array(y_out.persister_y[:, 0:p['n_strains']] > 0).sum(axis=1)
            frac = y_out.persister_y[:, 0:p['n_strains']] / np.maximum(np.tile(persister['Parasites'], (p['n_strains'], 1)).transpose(), 1e-10)
            persister['Diversity'] = -np.sum(frac * np.log(np.maximum(frac, 1e-10)), 1)
            persister['Evenness'] = persister['Diversity'] / np.log(np.maximum(persister['COI'], 2))
            persister['Cross immunity'] = y_out.persister_y[:, -1]
            persister['Persister'] = True

        non_persister = pd.DataFrame(columns=['Time', 'Parasites', 'COI', 'Diversity', 'Evenness', 'Cross immunity', 'Persister'])
        if len(y_out.non_persister_y) > 0:
            non_persister['Time'] = y_out.non_persister_t
            non_persister['Parasites'] = y_out.non_persister_y[:, 0:p['n_strains']].sum(axis=1)
            non_persister['COI'] = np.array(y_out.non_persister_y[:, 0:p['n_strains']] > 0).sum(axis=1)
            frac = y_out.non_persister_y[:, 0:p['n_strains']] / np.maximum(np.tile(non_persister['Parasites'], (p['n_strains'], 1)).transpose(), 1e-10)
            non_persister['Diversity'] = -np.sum(frac * np.log(np.maximum(frac, 1e-10)), 1)
            non_persister['Evenness'] = non_persister['Diversity'] / np.log(np.maximum(non_persister['COI'], 2))
            non_persister['Cross immunity'] = y_out.non_persister_y[:, -1]
            non_persister['Persister'] = False

        return y_out, pd.concat([persister.dropna(), non_persister.dropna()], ignore_index=True)
    else:
        return y_out


def params(mean_infections_per_season=3, t_dry=181, years=20, year_duration=365):
    """
    Returns default values constant values for the model in a dictionary.
    """
    params = {}
    strains_per_season = np.random.poisson(lam=mean_infections_per_season, size=years).astype(int)
    params['n_strains'] = strains_per_season.sum().astype(int) + 1
    params['t_0'] = 0
    params['t_f'] = year_duration * years
    params['t_den'] = 1.0
    params['t_dry'] = t_dry
    params['year_duration'] = year_duration
    params['inf_times'] = np.append(
        np.concatenate([
            np.sort(np.random.random(strains_per_season[y]) * t_dry + (y * year_duration))
            for y in range(years)
        ]),
        [year_duration * years]
    )
    params['sigma'] = 0
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

    # Parameters for the Hill function
    params['K_hill'] = 0.7  # Half-saturation constant
    params['n_hill'] = 3  # Hill coefficient

    return params


def solveModel(persister_out=True, num_simulations=32, **kwargs):
    '''
    Main method containing multiple solver calls and concatenation of results.
    Writes figures to file.
    '''
    all_results = []
    all_dfs = []

    for sim in range(num_simulations):
        p = params()
        
        p.update(kwargs)
        t = np.linspace(p['t_0'], p['t_f'], int((p['t_f'] - p['t_0']) / p['t_den']) + 1)


        y_0 = initCond(p)  # get initial conditions

        # Solve model
        sol, df = odeSolver(p['odeFun'], t, y_0, p, solver="RK45", persister_out=persister_out)

        # Assign simulation ID to the dataframe
        df['ID'] = sim

        all_results.append(sol)
        all_dfs.append(df)

 
    combined = pd.concat(all_dfs, ignore_index=True)
    
    combined= combined.groupby(["ID", "Persister"]).agg('mean').reset_index()

    

    return all_results, combined

# Definir los rangos de K_hill y n_hill
K_hill_values = np.linspace(0.5, 0.8, 4)  # [0.5, 0.6, 0.7, 0.8]
n_hill_values = np.arange(1, 6)           # [1, 2, 3, 4, 5]

# DataFrame para guardar los resultados medios
results = []

for K_hill in K_hill_values:
    for n_hill in n_hill_values:
        _, combined = solveModel(
            num_simulations=120,
            K_hill=K_hill,
            n_hill=n_hill
        )
        # Calcular la media para Persister True y False por separado
        for persister_value in [True, False]:
            subset = combined[combined['Persister'] == persister_value]
            if not subset.empty:
                means = subset.drop(columns=['ID', 'Persister']).mean()
                means['K_hill'] = K_hill
                means['n_hill'] = n_hill
                means['Persister'] = persister_value
                results.append(means)

# %%
df_means = pd.DataFrame(results)

# Valores experimentales de COI para Persister False y True
exp_coi = {False: 2.263159, True: 3.080460}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Evita el DeprecationWarning seleccionando solo la columna 'COI'
vmin = df_means.groupby('Persister')['COI'].apply(lambda s: np.abs(s - exp_coi[s.name])).min()
vmax = df_means.groupby('Persister')['COI'].apply(lambda s: np.abs(s - exp_coi[s.name])).max()

for idx, persister_value in enumerate([False, True]):
    df_means_p = df_means[df_means['Persister'] == persister_value].copy()
    df_means_p['diff'] = np.abs(df_means_p['COI'] - exp_coi[persister_value])
    heatmap_data = df_means_p.pivot(index='n_hill', columns='K_hill', values='diff')
    heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
    rounded_cols = [f"{col:.1g}" for col in heatmap_data.columns]
    sns.heatmap(
        heatmap_data,
        annot=False,
        cmap='magma',
        ax=axes[idx],
        cbar=idx == 1,
        fmt=".2f",
        vmin=vmin,
        vmax=vmax
    )
    axes[idx].set_title(f"Persistent infection = {persister_value}")
    axes[idx].set_xlabel('K [half-saturation of cross-immunity]', labelpad=15)
    axes[idx].set_ylabel('n [Hill coefficient ]', rotation=90, labelpad=15)  # Etiqueta rotada 90°
    axes[idx].yaxis.set_label_position("left")
    axes[idx].set_yticklabels(heatmap_data.index, rotation=0)  # Muestra los valores de n
    axes[idx].set_xticklabels(rounded_cols, rotation=0)

plt.tight_layout()
plt.show()

