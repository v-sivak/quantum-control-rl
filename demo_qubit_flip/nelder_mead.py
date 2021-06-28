# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:54:06 2021

@author: qulab
"""
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
from scipy.optimize import minimize

def fidelity_estimator(a, M, _type='collection'):
    scale = 2*np.pi
    theta = a * scale
    prob = np.sin(theta/2)**2
    if _type=='collection':
        P = tfp.distributions.Bernoulli(probs=[prob]*M)
        sigma_z = 1 - 2 * P.sample()
        rewards = -np.mean(sigma_z)
    elif _type=='evaluation':
        sigma_z = 1 - 2 * prob
        rewards = -sigma_z
    return rewards


def cost(a):
    print('Trying a=%.3f' %a[0])
    f = fidelity_estimator(a[0], M=180)
    print('Got fidelity =%.6f' %f)
    print('')
    return -f

all_F = []

def callback(x):
    print(x)

for SEED in range(100):
    init_simplex = np.random.normal(loc=0, scale=0.5, size=[2,1])
    
    res = minimize(cost, np.array(0.2), method='Nelder-Mead',
                   options=dict(maxfev=10, initial_simplex=init_simplex, return_all=True))
    
    
    F = fidelity_estimator(res.x, 1, _type='evaluation')
    print('')
    print('F=%.5f'%F)
    all_F.append(F)
    
print(np.median(all_F))