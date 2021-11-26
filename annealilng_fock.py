# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:54:06 2021
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize, Bounds, dual_annealing
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc
from math import sqrt, pi
import qutip as qt

class ActionScript():
    pass

for sigma in [0.6]:
    for fock in range(1,11):
        for seed in range(6):
            
            np.random.seed(seed)
    
            N_iter = 10000
            batch_size = 1
            T = 5
            N = 100
            action_scale = {'alpha':4, 'theta':pi}
            action_size = {'alpha':[T,2], 'theta':[T,15]}
            
            
            # target_state = qt.basis(N,fock)
            # reward_kwargs = {'reward_mode' : 'fock',
            #                   'target_state' : target_state,
            #                   'N_msmt' : 2000,
            #                   'error_prob' : 0}

            target_state = qt.basis(N,fock)
            reward_kwargs = {'reward_mode' : 'overlap',
                              'target_state' : target_state,
                              'postselect_0' : False}
            
            env = env_init(control_circuit='snap_and_displacement', reward_kwargs=reward_kwargs,
                           init='vac', T=T, batch_size=batch_size, N=N, episode_length=T)
            
            
            def avg_reward(action_script):
            
                policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)
                
                time_step = env.reset()
                policy_state = policy.get_initial_state(env.batch_size)
                while not time_step.is_last()[0]:
                    action_step = policy.action(time_step, policy_state)
                    policy_state = action_step.state
                    time_step = env.step(action_step.action)
                
                return time_step.reward.numpy().mean()
            
            
            N_alpha = np.prod(action_size['alpha'])
            N_theta = np.prod(action_size['theta'])
            N_tot = N_alpha + N_theta
            
            class Optimization():
                
                def __init__(self):
                    self.all_actions = {'alpha': [], 'theta':[]}
                    self.all_costs = []
                    self.cost_evals = 0 
                
                def cost(self, a):
                    self.cost_evals += 1
                    
                    action_script = ActionScript()
                    action_script.period = T
                    action_script.script = {}
                    action_script.script['alpha'] = a[:N_alpha].reshape(action_size['alpha']) * action_scale['alpha']
                    action_script.script['theta'] = a[N_alpha:].reshape(action_size['theta']) * action_scale['theta']
                    
                    self.all_actions['alpha'].append(action_script.script['alpha'])
                    self.all_actions['theta'].append(action_script.script['theta'])
                    
                    S = avg_reward(action_script)
                    self.all_costs.append(-S)
                    
                    return -S
                
                def export_all_actions(self, savedir):
                    np.savez(savedir, alpha=np.array(self.all_actions['alpha']),
                             theta=np.array(self.all_actions['theta']),
                             cost=np.array(self.all_costs))
            
            opt = Optimization()
            

            x0 = np.random.uniform(low=-1, high=1, size=[N_tot])*sigma
            res = dual_annealing(opt.cost, [[-1,1]]*N_tot, maxfun=N_iter, x0=x0, 
                                 no_local_search=True, visit=1.1, initial_temp=0.1)
            
            nmdir = r'E:\data\gkp_sims\annealing\8'
            fname = 'fock_' + str(fock) + '_seed_' + str(seed) + '_sigma_' + str(sigma) + '.npz'
            opt.export_all_actions(os.path.join(nmdir,fname))
