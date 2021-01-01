# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:16:23 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
import tensorflow_probability as tfp

actions = ['alpha']#, 'phi_e', 'phi_g']


# trainable variables
mean = {s : tf.Variable(tf.random.uniform([]), name='mean_'+s) for s in actions}
sigma = {s : tf.Variable(0.5, name='sigma_'+s) for s in actions}
baseline = tf.Variable(0.0, name='baseline')


algo = 'PPO'
B = 200 # batch_size
EPOCHS = 1000
lr = 1e-4
policy_steps = 20

def reward_sampler(a):
    p = 1/(1+500*(a['alpha'] - 0.37)**2) # just needed a sharper reward funciton!
    z = tfp.distributions.Bernoulli(probs=p).sample()
    return 2*tf.cast(z, tf.float32)-1


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
    mean_old = tf.nest.map_structure(tf.identity, mean)
    sigma_old = tf.nest.map_structure(tf.identity, sigma)
    
    for i in range(policy_steps):
        # calculate policy loss and do gradient update
        with tf.GradientTape() as tape:    
            # log prob according to the current policy
            log_prob = compute_log_prob(a, mean, sigma)
    
            A = R - baseline # action advantages       
    
            if algo == 'REINFORCE':
                policy_loss_batch = - A * log_prob
    
            if algo == 'PPO':
                eps = 0.2
                importance_ratio = tf.math.exp(log_prob - log_prob_old)
                importance_ratio_clip = tf.clip_by_value(importance_ratio, 1-eps, 1+eps)
                policy_loss_batch = - tf.minimum(importance_ratio*A, importance_ratio_clip*A)
            
            policy_loss = tf.reduce_mean(policy_loss_batch) # reduce over batch
            value_loss = tf.reduce_mean(A**2)
            loss = policy_loss + value_loss
            
            grads = tape.gradient(loss, tape.watched_variables())
            optimizer.apply_gradients(zip(grads, tape.watched_variables()))
            
    eval_action = tf.nest.map_structure(tf.math.tanh, mean)
    print(eval_action['alpha'])
        


