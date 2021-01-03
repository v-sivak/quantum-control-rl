# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:16:23 2020

@author: Vladimir Sivak

This is a straightforward implementation of PPO http://arxiv.org/abs/1707.06347
for simple single-state RL environments.
"""
import tensorflow as tf
import tensorflow_probability as tfp
from remote_env_tools.remote_env_tools import Server


# initialize the "agent server" and connect to "environment client"
server_socket = Server()
(host, port) = '127.0.0.1', 5000
server_socket.bind((host, port))
server_socket.connect_client()

def reward_sampler(actions):
    server_socket.send_data(actions)
    rewards, done = server_socket.recv_data()
    return rewards


# trainable variables
actions = ['alpha', 'phi_e', 'phi_g']
mean = {s : tf.Variable(tf.random.normal([], stddev=0.05), name='mean_'+s) for s in actions}
sigma = {s : tf.Variable(0.5, name='sigma_'+s) for s in actions}
baseline = tf.Variable(0.0, name='baseline')


algo = 'PPO'
B = 200 # batch_size
EPOCHS = 3000
lr = 1e-3
policy_steps = 20

log_prob_clipping = 10.
gradient_clipping = 1.
importance_ratio_eps = 0.2


def compute_log_prob(action, mean, sigma):
    sigma_eps = 1e-5 # for mumerical stability
    log_prob = 0.
    for s in action.keys():
        log_prob += - tf.math.log(tf.math.abs(sigma[s]) + sigma_eps) \
            - 0.5 * (action[s] - mean[s])**2 / (sigma[s]**2 + sigma_eps)
    return log_prob

optimizer = tf.optimizers.Adam(learning_rate=lr)

for epoch in range(EPOCHS):
    # sample a batch of actions from Gaussian policy
    N = {s : tfp.distributions.Normal(loc=mean[s], scale=sigma[s]) for s in actions}
    a = {s : N[s].sample(B) for s in actions}
    a_tanh = tf.nest.map_structure(tf.math.tanh, a)

    R =  reward_sampler(a_tanh) # collect rewards
    
    # log prob according to old policy (required for importance ratio)
    if epoch == 0: mean_old, sigma_old = mean, sigma 
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
                importance_ratio_clip = tf.clip_by_value(importance_ratio, 1-importance_ratio_eps, 1+importance_ratio_eps)
                policy_loss_batch = - tf.minimum(importance_ratio*A, importance_ratio_clip*A)
            
            policy_loss = tf.reduce_mean(policy_loss_batch) # reduce over batch
            value_loss = tf.reduce_mean(A**2)
            loss = policy_loss + value_loss
            
            grads = tape.gradient(loss, tape.watched_variables())
            grads = tf.clip_by_value(grads, -gradient_clipping, gradient_clipping)
            optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            
    eval_action = tf.nest.map_structure(tf.math.tanh, mean)
    # if epoch % 10 == 0: 
    #     print(test_eval(eval_action))
    # print(eval_action['alpha'])
    print(float(tf.reduce_mean(R)))
        

server_socket.disconnect_client()