# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 10:16:53 2021

@author: qulab
"""
from fpga_lib.scripting import get_experiment, run_and_get_results
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(t, A, tau, ofs):
    return A * np.exp(-t/tau) + ofs


exp = get_experiment('gkp_exp.gkp_qec.logical_lifetime_save_msmts', from_gui=True)
steps = (0, 564, 48)
rounds = np.linspace(*steps, dtype=int)

all_results = []
for reps in rounds:
    print('Starting reps = %d' %reps)
    exp.set_params({'reps' : reps, 'n_blocks' : 200, 'averages_per_block' : 100})
    results = run_and_get_results(exp)
    all_results.append(results)
    



def filter_leakage_vs_outliers(a):
    cond1 = np.where(a==1, True, False)
    cond2 = np.where(np.roll(a,1,axis=1)==1, True, False)
    cond3 = np.where(np.roll(a,-1,axis=1)==1, True, False)
    
    cond = cond1 * cond2 + cond1 * cond3
    return cond.astype(int)

        
# Account for flipping of the logical Pauli by SBS step
# This mask assumes that protocol starts with 'x' quadrature
flip_mask = {'X' : np.array([+1,-1,-1,+1]*int(steps[1]))[rounds],
             'Y' : np.array([+1,+1,+1,+1]*int(steps[1]))[rounds],
             'Z' : np.array([+1,+1,-1,-1]*int(steps[1]))[rounds]}
        

all_m2_sz, logical_pauli, fit_params, rounds_filtered = {}, {}, {}, {}
leakage_fraction = {}

for s in ['X', 'Y', 'Z']:
    all_m2_sz[s] = []
    leakage_fraction[s] = [0.0]
    for i, reps in enumerate(rounds):
        
        # first post-select on the correct initialization (qubit ending in 'g')
        init_state_g = all_results[i]['m0_'+s].multithresh()[0]
        postselected = all_results[i]['m2_'+s].postselect(init_state_g, 1)
        
        # then post-select on not having any leakage events at any step 
        if reps != 0:
#            mask = np.any(all_results[i]['mi_'+s].multithresh()[2].data, axis=1).astype(int)
            outliers = filter_leakage_vs_outliers(all_results[i]['mi_'+s].multithresh()[2].data)
            mask = np.any(outliers, axis=1).astype(int)
            
            leakage_fraction[s].append(np.mean(mask))
            postselected = postselected.postselect(mask, 0)
        
        m2_sz = 1 - 2*postselected.thresh_mean().data
        all_m2_sz[s].append(m2_sz)
    
    # Every other round mixer phase is in the wrong quadrature for X and Z
    # so we need to select only part of the data
    ind = np.where(rounds % 2 == 0)
    logical_pauli[s] = (np.array(all_m2_sz[s]) * flip_mask[s])[ind]
    rounds_filtered[s] = rounds[ind]
    
    # Sometimes leakage post-selection creates NaNs, filter those out too.
    ind_nan = np.where(np.isnan(logical_pauli[s])==False)
    logical_pauli[s] = logical_pauli[s][ind_nan]
    rounds_filtered[s] = rounds_filtered[s][ind_nan]
    
    # fit exponential and clean up large ouliers (due to some fpga error)
    p0 = (1,100,0)
    popt, pcov = curve_fit(exp_decay, rounds_filtered[s], logical_pauli[s], p0=p0)
    sigma = np.abs(logical_pauli[s] - exp_decay(rounds_filtered[s], *popt))
    mask = sigma > 3*np.mean(sigma)
    logical_pauli[s] = logical_pauli[s][np.where(mask==False)]
    rounds_filtered[s] = rounds_filtered[s][np.where(mask==False)]
    
    # fit again after removing the outliers
    popt, pcov = curve_fit(exp_decay, rounds_filtered[s], logical_pauli[s], p0=p0)
    fit_params['logical_'+s+':A'] = popt[0]
    fit_params['logical_'+s+':tau'] = popt[1]
    fit_params['logical_'+s+':ofs'] = popt[2]


# Plot logical lifetimes
fig, ax = plt.subplots(1,1)
ax.set_xlabel('Round')
colors = {'X': 'blue', 'Y': 'red', 'Z': 'green'}
for s in ['X','Y','Z']:
    ax.plot(rounds_filtered[s], logical_pauli[s],
            linestyle='none', marker='.', color=colors[s],
            label=s+', T=%.0f' % fit_params['logical_'+s+':tau'])
    ax.plot(rounds_filtered[s], exp_decay(rounds_filtered[s], 
            fit_params['logical_'+s+':A'], fit_params['logical_'+s+':tau'], fit_params['logical_'+s+':ofs']),
            linestyle='-', color=colors[s])
ax.grid(True)
ax.legend()
plt.tight_layout()

# Plot fraction of trajectories that experienced leakage
fig, ax = plt.subplots(1,1)
ax.set_title('Fraction of trajectories that experienced leakage')
for s in ['X','Y','Z']:
    leakage_fraction[s] = np.array(leakage_fraction[s])
    popt, pcov = curve_fit(exp_decay, rounds, 1-leakage_fraction[s], p0=(1,100,0))
    ax.plot(rounds, leakage_fraction[s], marker='.', linestyle='none', color=colors[s])
    ax.plot(rounds, 1-exp_decay(rounds, *popt), color=colors[s],
            label=s+', T=%.0f' % popt[1])
ax.grid(True)
ax.legend()
plt.tight_layout()


# Plot leakage events 
fig, ax = plt.subplots(1,1, dpi=200)
this_data = filter_leakage_vs_outliers(all_results[-1]['mi_'+s].multithresh()[2].data)[:1000]
vals = np.where(this_data==1, -0.9, 0.6)
ax.pcolormesh(np.transpose(vals), cmap='RdYlGn', vmin=-1, vmax=1)
#ax.set_aspect('equal')
ax.set_ylabel('Time-step')
ax.set_xlabel('Episode')
plt.tight_layout()
