import numpy as np
import pandas as pd
from ODE_Solver_persistence import params, shannon_diversity
from utils import sweep_parameters, runs
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

p= params()

#Hypothesis: this should not effect on persistency  
     
# First simulation: r is a value in a uniform distribution between 0.5 to 1
r_values_uniform = np.sort(np.random.uniform(0.5, np.log(14)/2, 10))
s1= sweep_parameters(p,'r',r_values_uniform)
    
# Second simulation: r is a randomly picked value in the same uniform distribution for each n_strain
r_values= np.random.uniform(0.5, np.log(14) / 2, p['n_strains'])
p1=p.copy()
p1['r'] = r_values
s2= runs(p1,10)

#Third simulation: alpha is a value between 7e-4 to 7e-8 
exponents = np.arange(4, 8, 0.5)
alpha_values = 7 * 10**(-exponents)
s3= sweep_parameters(p,'a',alpha_values)
    
#Analysis 


parasites_all = []
diversity_all = []
evenness_all = []
cross_im_all=[]

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

for i in range(len(s1)):
    sol = s1[i][1]
    parasites, cross_im, coi, diversity, evenness = shannon_diversity(sol, p['n_strains'])

    parasites_all.append(parasites)
    diversity_all.append(diversity)
    evenness_all.append(evenness)
    cross_im_all.append(cross_im)

    time = sol.t

    axs[0].plot(time, parasites)
    axs[0].set_yscale('log')
    axs[1].plot(time, cross_im)
    axs[2].plot(time, coi)
    axs[3].plot(time, diversity)
    axs[4].plot(time, evenness)

axs[0].set_ylabel('Parasites')
axs[1].set_ylabel('Cross Immunity')
axs[2].set_ylabel('COI')
axs[3].set_ylabel('Diversity')
axs[4].set_ylabel('Evenness')
axs[4].set_xlabel('Time')


plt.show()


fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
for i in range(len(s2)):
    sol = s2[i][1]
    parasites, cross_im, coi, diversity, evenness = shannon_diversity(sol, p['n_strains'])

    parasites_all.append(parasites)
    diversity_all.append(diversity)
    evenness_all.append(evenness)
    cross_im_all.append(cross_im)

    time = sol.t

    axs[0].plot(time, parasites)
    axs[0].set_yscale('log')
    axs[1].plot(time, cross_im)
    axs[2].plot(time, coi)
    axs[3].plot(time, diversity)
    axs[4].plot(time, evenness)

axs[0].set_ylabel('Parasites')
axs[1].set_ylabel('Cross Immunity')
axs[2].set_ylabel('COI')
axs[3].set_ylabel('Diversity')
axs[4].set_ylabel('Evenness')

fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
colors = plt.cm.tab10(np.linspace(0, 1, len(s3)))  # Use a discrete colormap

for i in range(len(s3)):
    sol = s3[i][1]
    parasites, cross_im, coi, diversity, evenness = shannon_diversity(sol, p['n_strains'])

    parasites_all.append(parasites)
    diversity_all.append(diversity)
    evenness_all.append(evenness)
    cross_im_all.append(cross_im)

    time = sol.t

    color = colors[i % len(colors)]  # Cycle through the discrete colors
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
handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2) for i in range(len(s3))]
labels = [f'a={alpha_values[i]:.1e}' for i in range(len(s3))]
axs[0].legend(handles, labels, loc='center right', bbox_to_anchor=(1.25, 0.5), frameon=False)

plt.show()