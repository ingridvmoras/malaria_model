
import numpy as np
import pandas as pd
import persistance
from utils import sweep_parameters, runs, params,sweep_mean_infections,shannon_diversity,runs_a, sweep_inf_alpha
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import plots
import seaborn as sns

np.random.seed(19680801)

p= params()

# #Hypothesis: this should not effect on persistency  
     
# # First simulation: r is a value in a uniform distribution between 0.5 to 1
# r_values_uniform = np.log(np.sort(np.random.uniform(1, 14, 10)))/2
# print('Calculating simulation 1')
# s1= sweep_parameters(p,'r',r_values_uniform)

# # Second simulation: r is a randomly picked value in the same uniform distribution for each n_strain
# print('Calculating simulation 2')
# s2= runs('r',p,10)

# #Third simulation: alpha is a value between 7e- 5to 7e-7 
# exponents = np.arange(4, 8, 0.25)
# alpha_values = 7 * 10**(-exponents)
# print('Calculating simulation 3')
# s3= sweep_parameters(p,'a',alpha_values)


# #Fourth simulation: alpha is randomly picked value from a normal distribution for each n_strain

print('Calculating simulation 4')
# s41= runs_a(p,10) #uniforme
s4,distributions= runs_a(p,120,'normal',[1*(10**-6)],persister_out=True)
sd_values = np.linspace( 1 * (10**-6)- (0.95 *1 * (10**-6) ), 1 * (10**-6)+ 1 * (10**-6), 120)
s4_2, distributions2 = runs_a(p, 120, 'multiNorm', sd_values, persister_out=True)

# Combine all dataframes into one dataframe with an ID column

s4_2df = pd.DataFrame()

for idx, (sd, df) in enumerate(s4_2):
    df = df.copy()
    df['sd'] = sd
    df['ID'] = idx + 1  
    s4_2df = pd.concat([s4_2df, df], ignore_index=True)
    
s4_df = pd.DataFrame()

for idx, (id, df) in enumerate(s4):
    df = df.copy()
    df['ID'] = idx + 1
    s4_df = pd.concat([s4_df, df], ignore_index=True)



plots.figDistributions(s4_2df,p,f_name='.\\plots\\ODE_persistence_s4_2.png')
plots.figDistributions(s4_df,p,f_name='.\\plots\\ODE_persistence_s4.png')




#Changing mean infection times 

#mean_infections=np.round(np.linspace(1,10,10))
#print('Calculating simulation 5')
#s5=sweep_mean_infections(mean_infections)

# Plot the simulations
# df1=plots.plot_simulations(s1, p, 'r', r_values_uniform, f_name='simulation_plot_s1.png')
# df2=plots.plot_simulations(s2, p, 'r', [s2[i][0] for i in range(len(s2))], f_name='simulation_plot_s2.png')
# df3=plots.plot_simulations(s3, p, 'a', alpha_values, f_name='simulation_plot_s3.png')
#df4=plots.plot_simulations(s4, p, 'a', [s4[i][0] for i in range(len(s4))], f_name='simulation_plot_s4.png')
# df5=plots.plot_simulations(s41, p, 'a', [s41[i][0] for i in range(len(s41))], f_name='simulation_plot_s41.png')
#df6=plots.plot_simulations(s42, p, 'sd', sd_values, f_name='simulation_plot_s42.png')
#df7=plots.plot_simulations(s5, p, 'mean infections', mean_infections, f_name='simulation_plot_s5.png')


# parasites,cross_im,coi,shannon_diversity,shannon_evenness= shannon_diversity(s4[5][1],p['n_strains'])
# plots.figTSeries(s4[5][1],coi, shannon_diversity, shannon_evenness, p)
#distributions['simulation'] = round(distributions['simulation'])
#plots.normal_distributions(distributions)





