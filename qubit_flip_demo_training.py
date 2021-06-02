# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:12:45 2021

@author: Vladimir Sivak
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

rootdir = r'E:\data\gkp_sims\PPO\simple_demo\sweep11'

for SEED in range(100):

    # trainable variables
    actions = ['theta']
    mean = {s : tf.Variable(tf.random.normal([], stddev=0.05), name='mean_'+s) for s in actions}
    sigma = {s : tf.Variable(0.5, name='sigma_'+s) for s in actions}
    baseline = tf.Variable(0.0, name='baseline')
    
    
    algo = 'PPO'
    B = 30 # batch_size
    EPOCHS = 100
    eval_interval = 1
    lr = 5e-3 # 1e-2 for SGD
    policy_steps = 20
    
    log_prob_clipping = 10.
    gradient_clipping = 1.
    importance_ratio_eps = 0.20
    value_loss_coeff = 0.5
    
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    # optimizer = tf.optimizers.SGD(learning_rate=lr)
    
    def reward_sampler(a, epoch, _type):
        """
        This implements a unitary U(a) = exp(-i*pi*a*sigma_x) on the state |0>,
        and then simply samples sigma_z measurement outcomes with the resulting
        state, and returns -sigma_z as a reward. THis will encaurage the agent
        to prepare |1> state. The ideal action would correspond to a=0.5
        """
        a_np = {s : np.array(a[s] if _type=='collection' else [a[s]]) for s in actions}
        
        scale = 2*np.pi
        theta = a_np['theta'] * scale
        probs = np.sin(theta/2)**2
        if _type=='collection':
            P = tfp.distributions.Bernoulli(probs=probs)
            sigma_z = 1 - 2 * P.sample()
            rewards = -sigma_z
        elif _type=='evaluation':
            sigma_z = 1 - 2 * probs
            print('EVAL NOW %.3f' %float(probs))
            print('EVAL NOW %.3f' %float(sigma_z))
            rewards = -sigma_z
        return tf.cast(rewards, tf.float32)
    
    
    def evaluation(epoch, log=None):
        value = lambda a: a.value()
        eval_action = tf.nest.map_structure(value, mean)
        R_eval = reward_sampler(eval_action, epoch, 'evaluation')
        print('Deterministic policy: %.3f' %float(tf.reduce_mean(R_eval)))
        for s in actions:
            train_vars_tuple = (float(mean[s].numpy()), float(sigma[s].numpy()))
            print(s+': %.5f +- %.5f' %train_vars_tuple)
        if log is not None:
            log['eval_rewards'].append(np.array(R_eval))
            log['eval_epochs'].append(epoch)
            log['mean'].append(mean['theta'].numpy())
            log['sigma'].append(sigma['theta'].numpy())
    
    def compute_log_prob(action, mean, sigma):
        sigma_eps = 1e-5 # for mumerical stability
        log_prob = 0.
        for s in action.keys():
            log_prob += - tf.math.log(tf.math.abs(sigma[s]) + sigma_eps) \
                - 0.5 * (action[s] - mean[s])**2 / (sigma[s]**2 + sigma_eps)
        return log_prob
    
    log = dict(train_rewards=[], train_epochs=[], eval_rewards=[], eval_epochs=[],
               train_actions=[], mean=[], sigma=[])
    
    evaluation(0, log) # evaluate policy once before training
    
    for epoch in range(1,EPOCHS+1):
        # sample a batch of actions from Gaussian policy
        N = {s : tfp.distributions.Normal(loc=mean[s], scale=sigma[s]) for s in actions}
        a = {s : N[s].sample(B) for s in actions}
    
        # collect those rewards
        R =  reward_sampler(a, epoch, 'collection')
        
        # log prob according to old policy (required for importance ratio)
        if epoch == 1: mean_old, sigma_old = mean, sigma 
        log_prob_old = compute_log_prob(a, mean_old, sigma_old)
        log_prob_old = tf.clip_by_value(log_prob_old, -log_prob_clipping, log_prob_clipping)
        mean_old = tf.nest.map_structure(tf.identity, mean)
        sigma_old = tf.nest.map_structure(tf.identity, sigma)
        
        # calculate policy loss and do several gradient updates
        for i in range(policy_steps):
            with tf.GradientTape() as tape:
                # log prob according to the current policy
                log_prob = compute_log_prob(a, mean, sigma)
                log_prob = tf.clip_by_value(log_prob, -log_prob_clipping, log_prob_clipping)
                
                A = R - baseline # action advantages       
        
                if algo == 'REINFORCE':
                    policy_loss_batch = - A * log_prob
                    
                if algo == 'PPO':
                    importance_ratio = tf.math.exp(log_prob - log_prob_old)
                    importance_ratio_clip = tf.clip_by_value(importance_ratio, 
                                1-importance_ratio_eps, 1+importance_ratio_eps)
                    policy_loss_batch = -tf.minimum(importance_ratio*A, importance_ratio_clip*A)
                
                policy_loss = tf.reduce_mean(policy_loss_batch) # reduce over batch
                value_loss = tf.reduce_mean(A**2)
                loss = policy_loss + value_loss_coeff * value_loss
                
                grads = tape.gradient(loss, tape.watched_variables())
                grads = tf.clip_by_value(grads, -gradient_clipping, gradient_clipping)
                optimizer.apply_gradients(zip(grads, tape.watched_variables()))
        print('Epoch %d: %.3f' %(epoch, float(tf.reduce_mean(R))))
        log['train_rewards'].append(np.array(R))
        log['train_actions'].append(np.array(a['theta']))
        log['train_epochs'].append(epoch)
        if epoch % eval_interval == 0: evaluation(epoch, log)
    
    
    # # Plot training progress
    # from plotting import plot_config
    # fix, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=200)
    # plt.grid()
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Reward')
    # ax.set_ylim(-1,1)
    # R_mean = np.mean(log['train_rewards'], axis=1)
    # ax.plot(log['train_epochs'], R_mean, label='Stochastic policy')
    # ax.plot(log['eval_epochs'], log['eval_rewards'], label='Deterministic policy')
    # ax.legend(loc='best')
    # plt.tight_layout()
    
    
    # # Plot gaussian policies
    # def gaussian(x, _mean, _sigma):
    #     return 1/np.sqrt(2*np.pi*_sigma**2) * np.exp(-(x-_mean)**2/2/_sigma**2)
    
    # fig, ax = plt.subplots(1, 1, figsize=(3.375,2), dpi=200)
    # xs = np.linspace(-1, 1, 201)
    # for i in range(len(log['mean'])):
    #     ax.plot(xs, gaussian(xs, log['mean'][i], log['sigma'][i]))
    
    filename = os.path.join(rootdir, 'data'+str(SEED)+'.npz')
    np.savez(filename, **log)