
import numpy as np
import pandas as pd
from utils import sweep_parameters, runs, params,sweep_mean_infections,shannon_diversity,runs_a
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import plots

np.random.seed(19680801)

p= params()

#Hypothesis: this should not effect on persistency  
     
# First simulation: r is a value in a uniform distribution between 0.5 to 1
r_values_uniform = np.log(np.sort(np.random.uniform(1, 14, 10)))/2
s1= sweep_parameters(p,'r',r_values_uniform)
    
# Second simulation: r is a randomly picked value in the same uniform distribution for each n_strain

s2= runs('r',p,10)

#Third simulation: alpha is a value between 7e- 5to 7e-7 
exponents = np.arange(4, 8, 0.25)
alpha_values = 7 * 10**(-exponents)
s3= sweep_parameters(p,'a',alpha_values)


#Fourth simulation: alpha is randomly picked value from a normal distribution for each n_strain
s41= runs_a(p,10) #uniforme
s4,distributions= runs_a(p,10,'normal',1*(10**-4))


#Changing mean infection times 

mean_infections=np.round(np.linspace(1,10,10))
s5=sweep_mean_infections(mean_infections)

# Plot the simulations
plots.plot_simulations(s1, p, 'r', r_values_uniform, f_name='simulation_plot_s1.png')
plots.plot_simulations(s2, p, 'r', [s2[i][0] for i in range(len(s2))], f_name='simulation_plot_s2.png')
plots.plot_simulations(s3, p, 'a', alpha_values, f_name='simulation_plot_s3.png')
plots.plot_simulations(s4, p, 'a', [s4[i][0] for i in range(len(s4))], f_name='simulation_plot_s4.png')
plots.plot_simulations(s41, p, 'a', [s41[i][0] for i in range(len(s41))], f_name='simulation_plot_s41.png')
plots.plot_simulations(s5, p, 'mean infections', mean_infections, f_name='simulation_plot_s5.png')
#Analysis 

parasites,cross_im,coi,shannon_diversity,shannon_evenness= shannon_diversity(s4[5][1],p['n_strains'])
plots.figTSeries(s4[5][1],coi, shannon_diversity, shannon_evenness, p)
plots.normal_distributions(distributions)