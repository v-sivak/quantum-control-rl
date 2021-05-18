# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:11:33 2020

@author: Vladimir Sivak
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os  
import policy as plc
from tf_env import GKP
"""
Adapted from:
https://fairyonice.github.io/Extract-weights-from-Keras's-LSTM-and-calcualte-hidden-and-cell-states.html

"""


def LSTMlayer(weight, x_t, h_tm1, c_tm1):
    '''
    weight must be output of LSTM's layer.get_weights()
    
    c_tm1 -- Cell state at time t-1; shape=(1,units)
    h_tm1 -- Hidden state at time t-1; shape=(1,units)
    x_t -- Input at time t; shape=(1,nfeatures)
    
    W -- weights for input; shape=(nfeatures,units*4)
    U -- weights for hidden state; shape=(units,units*4)
    b -- biases; shape=(units*4,)
    
    '''
    def sigmoid(x):
        return(1.0/(1.0+np.exp(-x)))

    W, U, b = weight
    units = U.shape[0]
    s_t = (x_t.dot(W) + h_tm1.dot(U) + b)
    i  = sigmoid(s_t[:,:units])
    f  = sigmoid(s_t[:,1*units:2*units])
    _c = np.tanh(s_t[:,2*units:3*units])
    o  = sigmoid(s_t[:,3*units:])
    c_t = i*_c + f*c_tm1
    h_t = o*np.tanh(c_t)
    return h_t, c_t, i, f, o


if __name__ == '__main__':        
        
    ### Parameters
    episodes = 1000 #100000
    steps = 40
    horizon = 1
    
    ### Environment, policy, and training data
    action_wrapper = plc.ActionWrapperV1() 
    env = GKP(action_wrapper, 
              H=horizon, 
              episode_length=steps,
              batch_size=episodes,
              init='vac')
    
    policy = plc.MarkovianPolicyV2(env.time_step_spec())

    ### Generate episodes
    history = []
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    for t in range(steps+1):
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        stacked_obs = tf.concat([time_step.observation['action'],
                                  time_step.observation['msmt']], axis=2)
        history.append(stacked_obs)
        time_step = env.step(action_step.action)
    
    history = tf.concat(history, axis=1).numpy()
    meas = history[:,:-1,[-1]]
    

    ### Load the model
    sim_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims'
    name = 'Benchmarking_HybridMarkovian4Rounds\supervised_lstm\lstm.hdf5'
    model = keras.models.load_model(os.path.join(sim_dir,name))
    for layer in model.layers:
        if "LSTM" in str(layer):
            weightLSTM = layer.get_weights()
            units = weightLSTM[1].shape[0]
    
    ### Calculate activations manually
    names = ['Cell state', 'Hidden state', 'Input gate', 'Forget gate', 'Output gate']
    c = np.zeros((episodes,steps,units))
    h = np.zeros((episodes,steps,units))
    i = np.zeros((episodes,steps,units))
    f = np.zeros((episodes,steps,units))
    o = np.zeros((episodes,steps,units))
    for t in range(1, steps):
        h[:,t,:], c[:,t,:], i[:,t,:], f[:,t,:], o[:,t,:] = \
            LSTMlayer(weightLSTM, history[:,t,:], h[:,t-1,:], c[:,t-1,:])
    

    ### Plot mean activation and correlation with measurements for the gates
    cmaps = ['BrBG'] * 2 + ['binary'] * 3 
    vmin = [None, -1, 0, 0, 0]
    vmax = [None, +1, 1, 1, 1]
    for k, act in zip(range(5), [c,h,i,f,o]):
        fig, axes = plt.subplots(1,2, sharey=True, figsize=(12,5))
        axes[0].set_xlabel('Time steps')
        axes[0].set_ylabel('Units')
        axes[0].set_title('Mean ' + names[k])
        axes[0].pcolormesh(act.mean(axis=0).transpose(), cmap=cmaps[k],
                           vmin=vmin[k],vmax=vmax[k])

        corr = lambda x, y : (x*y).mean(axis=0) - x.mean(axis=0)*y.mean(axis=0)
        r = corr(act,meas)/np.sqrt(corr(act,act)*corr(meas,meas))
        
        axes[1].set_title('Correlation with measurements')
        axes[1].set_xlabel('Time steps')
        axes[1].pcolormesh(r.transpose(), cmap='RdBu_r', vmin=-1.5, vmax=+1.5)
        plt.tight_layout()
         
