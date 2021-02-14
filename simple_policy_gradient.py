# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:16:23 2020

@author: Vladimir Sivak

This is a straightforward implementation of PPO http://arxiv.org/abs/1707.06347
for simple single-state RL environments.
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow_probability as tfp
from remote_env_tools.remote_env_tools import Server
import numpy as np
import matplotlib.pyplot as plt

# initialize the "agent server" and connect to "environment client"
server_socket = Server()
(host, port) = '172.28.142.46', 5555
server_socket.bind((host, port))
server_socket.connect_client()


# trainable variables
actions = ['alpha']
mean = {s : tf.Variable(tf.random.normal([], stddev=0.05), name='mean_'+s) for s in actions}
sigma = {s : tf.Variable(0.5, name='sigma_'+s) for s in actions}
baseline = tf.Variable(0.0, name='baseline')


algo = 'PPO'
B = 40 # batch_size
reps = 5
eval_reps = 1000
EPOCHS = 100
eval_interval = 10
lr = 1e-3
policy_steps = 20

log_prob_clipping = 10.
gradient_clipping = 1.
importance_ratio_eps = 0.2

optimizer = tf.optimizers.Adam(learning_rate=lr)

def reward_sampler(a, epoch, reps, type='collection', compile_flag=True):
    a_np = {s : np.array(a[s] if type=='collection' else [a[s]]) for s in actions}
    batch_size = B if type=='collection' else 1
    data = {'type':type, 
            'epoch':epoch, 
            'action':a_np, 
            'reps':reps, 
            'batch_size':batch_size,
            'compile_flag': compile_flag}
    server_socket.send_data(data)
    rewards, done = server_socket.recv_data()
    return tf.cast(rewards, tf.float32)

def evaluation(epoch, log=None):
    eval_action = tf.nest.map_structure(tf.math.tanh, mean)
    R_eval = reward_sampler(eval_action, epoch, eval_reps, 'evaluation', True)
    print('Deterministic policy: %.3f' %float(tf.reduce_mean(R_eval)))
    for s in actions:
        train_vars_tuple = (float(mean[s].numpy()), float(sigma[s].numpy()))
        print(s+': %.5f +- %.5f' %train_vars_tuple)
    if log is not None:
        log['eval_rewards'].append(np.array(R_eval))
        log['eval_epochs'].append(epoch)

def compute_log_prob(action, mean, sigma):
    sigma_eps = 1e-5 # for mumerical stability
    log_prob = 0.
    for s in action.keys():
        log_prob += - tf.math.log(tf.math.abs(sigma[s]) + sigma_eps) \
            - 0.5 * (action[s] - mean[s])**2 / (sigma[s]**2 + sigma_eps)
    return log_prob

log = dict(train_rewards=[], train_epochs=[], eval_rewards=[], eval_epochs=[])

evaluation(0, log) # evaluate policy once before training
for epoch in range(1,EPOCHS+1):
    # sample a batch of actions from Gaussian policy
    N = {s : tfp.distributions.Normal(loc=mean[s], scale=sigma[s]) for s in actions}
    a = {s : N[s].sample(B) for s in actions}
    a_tanh = tf.nest.map_structure(tf.math.tanh, a)

    # collect those rewards (need to re-compile after evaluation rounds)
    compile_flag = True if epoch % eval_interval == 1 else False
    R =  reward_sampler(a_tanh, epoch, reps, 'collection', compile_flag)
    
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
            loss = policy_loss + value_loss
            
            grads = tape.gradient(loss, tape.watched_variables())
            grads = tf.clip_by_value(grads, -gradient_clipping, gradient_clipping)
            optimizer.apply_gradients(zip(grads, tape.watched_variables()))
    print('Epoch %d: %.3f' %(epoch, float(tf.reduce_mean(R))))
    log['train_rewards'].append(np.array(R))
    log['train_epochs'].append(epoch)
    if epoch % eval_interval == 0: evaluation(epoch, log)

server_socket.disconnect_client()


# plot training progress
from plotting import plot_config
fix, ax = plt.subplots(1,1, figsize=(3.375,2))
plt.grid()
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.set_ylim(-1,1)
R_mean = np.mean(log['train_rewards'], axis=1)
ax.plot(log['train_epochs'], R_mean, label='Stochastic policy')
ax.plot(log['eval_epochs'], log['eval_rewards'], label='Deterministic policy')
ax.legend(loc='best')
plt.tight_layout()