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

rootdir = r'E:\data\gkp_sims\PPO\simple_demo\sweep14'

for SEED in range(2000,3000,1):

    # trainable variables
    actions = ['theta']
    mean = {s : tf.Variable(tf.random.normal([], stddev=0.05), name='mean_'+s) for s in actions}
    sigma = {s : tf.Variable(0.5, name='sigma_'+s) for s in actions}
    baseline = tf.Variable(0.0, name='baseline')
    
    
    algo = 'PPO'
    # B = 5 # batch_size
    EPOCHS = 100
    eval_interval = 1
    lr = 1e-2 # 1e-2 for SGD
    policy_steps = 20
    
    log_prob_clip = 5
    grad_clip = 0.001
    importance_ratio_eps = 0.2
    value_loss_coeff = 0.5
    
    optimizer = tf.optimizers.Adam(learning_rate=lr)
    # optimizer = tf.optimizers.SGD(learning_rate=lr)


    def B_schedule(E):
        return 10

    
    # def B_schedule(E):
    #     if 0<E and E<=20:
    #         return 5
    #     if 20<E and E<=40:
    #         return 30
    
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
            log['baseline'].append(baseline.numpy())
    
    def compute_log_prob(action, mean, sigma):
        sigma_eps = 1e-5 # for mumerical stability
        log_prob = 0.
        for s in action.keys():
            log_prob += - tf.math.log(tf.math.abs(sigma[s]) + sigma_eps) \
                - 0.5 * (action[s] - mean[s])**2 / (sigma[s]**2 + sigma_eps)
        return log_prob
    
    log = dict(train_rewards=[], train_epochs=[], eval_rewards=[], eval_epochs=[],
               train_actions=[], mean=[], sigma=[], baseline=[], train_samples=[])
    
    evaluation(0, log) # evaluate policy once before training
    
    train_samples = 0
    
    for epoch in range(1,EPOCHS+1):
        
        B = B_schedule(epoch)
        train_samples += B
        
        # sample a batch of actions from Gaussian policy
        N = {s : tfp.distributions.Normal(loc=mean[s], scale=sigma[s]) for s in actions}
        a = {s : N[s].sample(B) for s in actions}
    
        # collect those rewards
        R =  reward_sampler(a, epoch, 'collection')
        
        # log prob according to old policy (required for importance ratio)
        if epoch == 1: mean_old, sigma_old = mean, sigma 
        log_prob_old = compute_log_prob(a, mean_old, sigma_old)
        log_prob_old = tf.clip_by_value(log_prob_old, -log_prob_clip, log_prob_clip)
        mean_old = tf.nest.map_structure(tf.identity, mean)
        sigma_old = tf.nest.map_structure(tf.identity, sigma)
        
        # calculate policy loss and do several gradient updates
        for i in range(policy_steps):
            with tf.GradientTape(persistent=True) as tape:
                # log prob according to the current policy
                log_prob = compute_log_prob(a, mean, sigma)
                log_prob = tf.clip_by_value(log_prob, -log_prob_clip, log_prob_clip)
                
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
                grads = tf.clip_by_value(grads, -grad_clip, grad_clip)
                optimizer.apply_gradients(zip(grads, tape.watched_variables()))


        print('Epoch %d: %.3f' %(epoch, float(tf.reduce_mean(R))))
        log['train_rewards'].append(np.array(R))
        log['train_actions'].append(np.array(a['theta']))
        log['train_epochs'].append(epoch)
        log['train_samples'].append(train_samples)
        if epoch % eval_interval == 0: evaluation(epoch, log)
    
    run_params = dict(B = B, EPOCHS = EPOCHS, eval_interval = eval_interval,
            lr = lr, policy_steps = policy_steps, log_prob_clip = log_prob_clip,
            grad_clip = grad_clip, importance_ratio_eps = importance_ratio_eps,
            value_loss_coeff = value_loss_coeff)
    
    filename = os.path.join(rootdir, 'data'+str(SEED)+'.npz')
    np.savez(filename, **log, **run_params)