import os
import seaborn as sns # for plots
from utils import shannon_diversity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d_name = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(d_name, exist_ok=True)

sns.set_style("ticks") # make pwetty plots
cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
              "#8E44AD", "#2ECC71"]
    # http://www.cookbook-r.com/Graphs/Colors_(ggplot2)/#a-colorblind-friendly-palette
    # http://jfly.iam.u-tokyo.ac.jp/color/

def figTSeries(sol, coi, diversity, evenness, params, f_name='ODE_tseries.png'):
    # Ensure the directory exists
    if os.path.dirname(f_name):
        os.makedirs(os.path.dirname(f_name), exist_ok=True)

    t = sol.t[:]  # get time values

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), dpi=200)  # make new figure

    n_strains = params['n_strains']
    cb_palette = ["#999999", "#E69F00", "#56B4E9", "#009E73",
                  "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

    # Plot Parasites
    ax = axs[0]
    for i in range(n_strains):
        ax.plot(t, sol.y[i, :], label='P' + str(i + 1), color=cb_palette[2], alpha=0.2)
    parasites = np.array(sol.y[0:n_strains, :].sum(axis=0))
    ax.plot(t, parasites, label='P', color=cb_palette[3])
    ax.set_yscale('log')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Parasites')

    # Plot Immune Response
    ax = axs[1]
    for i in range(n_strains):
        ax.plot(t, sol.y[n_strains + i, :], label='S' + str(i + 1), color=cb_palette[7], alpha=0.2)
    ax.plot(t, sol.y[-1, :], label='Cross Immunity', color=cb_palette[1])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Immune response')

    # Plot COI
    ax = axs[2]
    ax.plot(t, coi, label='COI', color=cb_palette[4])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('COI')

    # Plot Shannon Diversity and Evenness
    ax = axs[3]
    ax.plot(t, diversity, label='Shannon Diversity', color=cb_palette[6])
    ax.plot(t, evenness, label='Shannon Evenness', color=cb_palette[5])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Pathogens')

    # Combine all legends into one box on the right
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label in ['P', 'Cross Immunity', 'COI', 'Shannon Diversity', 'Shannon Evenness']:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.savefig(os.path.join(d_name, f_name), bbox_inches='tight')
    plt.show()


def figTSeries_r(sol, diversity, evenness, r_values, params, f_name='ODE_tseries.png'):
    # Ensure the directory exists
    if os.path.dirname(f_name):
        os.makedirs(os.path.dirname(f_name), exist_ok=True)

    t = sol.t[:]  # get time values

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), dpi=200)  # make new figure

    n_strains = params['n_strains']
    r_values = params['r']
    norm = plt.Normalize(vmin=min(r_values), vmax=max(r_values))
    cmap = plt.cm.viridis_r  # Inverted continuous color map

    # Plot Parasites
    ax = axs[0]
    for i in range(n_strains):
        color = cmap(norm(r_values[i]))
        ax.plot(t, sol.y[i, :], color=color, alpha=0.5)  # Increased alpha
    parasites = np.array(sol.y[0:n_strains, :].sum(axis=0))
    ax.plot(t, parasites, label='P', color=cb_palette[3], linewidth=2)  # Less transparent
    ax.set_yscale('log')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Parasites')
   
    # Plot Immune Response
    ax = axs[1]
    for i in range(n_strains):
        color = cmap(norm(r_values[i]))
        ax.plot(t, sol.y[n_strains + i, :], color=color, alpha=0.5)  # Increased alpha
    ax.plot(t, sol.y[-1, :], label='Cross Immunity', color=cb_palette[1], linewidth=2)  # Changed color
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Immune response')
    

    # Plot COI
    ax = axs[2]
    coi = np.array(sol.y[0:n_strains, :] > 0).sum(axis=0)
    ax.plot(t, coi, label='COI', color=cb_palette[4])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('COI')

    # Plot Shannon Diversity and Evenness
    ax = axs[3]

    shannon_diversity = diversity
    ax.plot(t, shannon_diversity, label='Shannon Diversity', color=cb_palette[6])
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Pathogens')

    shannon_evenness = evenness
    ax.plot(t, shannon_evenness, label='Shannon Evenness', color=cb_palette[7])
    

    # Add colorbar for r values outside the plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04, shrink=0.2)
    cbar.set_label('r', rotation=0, labelpad=10, ha='left')
    cbar.ax.tick_params(labelsize=8)  # Make the ticks smaller
    cbar.locator = plt.MaxNLocator(nbins=2)  # Reduce the number of intervals
    cbar.update_ticks()
    

    # Combine all legends into one box on the right
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.savefig(f_name, bbox_inches='tight')
    plt.show()
    
    
def plot_simulations(simulations, params, param_name, param_values, f_name='simulation_plot.png'):
    f_name = os.path.join(d_name, f_name)
    if os.path.dirname(f_name):
        os.makedirs(os.path.dirname(f_name), exist_ok=True)
    
    parasites_all = []
    diversity_all = []
    evenness_all = []
    cross_im_all = []
    coi_all=[]
    simulation=[]
    times=[]

    colors = plt.cm.viridis(np.linspace(0, 1, len(simulations)))  # Use the 'viridis' colormap
    _, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    

    for i in range(len(simulations)):
        sol = simulations[i][1]
        parasites, cross_im, coi, diversity, evenness = shannon_diversity(sol, params['n_strains'])
        for j in range(len(parasites)):
            parasites_all.append(parasites[j])
            diversity_all.append(diversity[j])
            evenness_all.append(evenness[j])
            cross_im_all.append(cross_im[j])
            coi_all.append(coi[j])
            simulation.append(simulations[i][0])
            times.append(sol.t[j])
        
        

        time = sol.t

        color = colors[i % len(colors)]  # Cycle through the colors
        axs[0].plot(time, parasites, color=color)
        axs[0].set_yscale('log')
        axs[1].plot(time, cross_im, color=color)
        axs[2].plot(time, coi, color=color)
        axs[3].plot(time, diversity, color=color)
        axs[4].plot(time, evenness, color=color)

    axs[0].set_ylabel('Parasites')
    axs[1].set_ylabel('Cross Immunity')
    axs[2].set_ylabel('COI')
    axs[3].set_ylabel('Diversity')
    axs[4].set_ylabel('Evenness')
    axs[4].set_xlabel('Time')

    # Add a legend
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(simulations))]
    labels = [f'{param_values[i]}' for i in range(len(simulations))] #:.2e
    axs[0].legend(handles, labels, loc='center right', bbox_to_anchor=(1.25, 0.5), frameon=False, title=param_name)
     
    parameter= [param_name]*len(parasites_all)
    
    df = pd.DataFrame({
        'parameter': [param_name] * len(parasites_all),
        'parasites': parasites_all,
        'diversity': diversity_all,
        'evenness': evenness_all,
        'cross_immunity': cross_im_all,
        'coi': coi_all, 
        'simulation': simulation,
        'times': times
    })
    
    plt.savefig(f_name, bbox_inches='tight')
    plt.show()
    return df
    
def normal_distributions(distributions):
    colors = plt.cm.viridis(np.linspace(0, 1, len(distributions['simulation'].unique()))).tolist()

    chart = sns.displot(data=distributions, x='a', hue='simulation', kind='kde', fill=False, palette=colors, height=5, aspect=1.5)
    
    chart._legend.set_title('Simulation')
    
    plt.savefig(os.path.join('plots\\normal_distributions.png'), bbox_inches='tight')
    plt.show()
    
    