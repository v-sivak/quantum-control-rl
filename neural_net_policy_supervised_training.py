# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:40:44 2020

@author: Vladimir Sivak
"""

import os
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="2"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tf_env import GKP
import policy as plc
from time import time

from IPython.display import clear_output

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation, Conv1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, \
    LearningRateScheduler, CSVLogger

class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0        
        self.fig = plt.figure()
        self.t = time()
        plt.suptitle('Loss (Mean Squared Error)')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        
    def on_epoch_end(self, epoch, logs={}):
        self.i += 1
        if self.i % 10 == 0:
            plt.plot([self.i], [logs.get('loss')], linestyle='none', 
                     marker='.', color='red')
            plt.pause(0.00001)
            print('============ Time: %f' %(time()-self.t))



def create_train_data(env, policy):
    observations, actions = [], []
    time_step = env.reset()
    policy_state = policy.get_initial_state(env.batch_size)
    while not time_step.is_last()[0]:
        action_step = policy.action(time_step, policy_state)      
        policy_state = action_step.state
        stacked_obs = tf.concat([time_step.observation['action'],
                                  time_step.observation['msmt']], axis=2)
        observations.append(stacked_obs)
        actions.append(action_step.action)
        time_step = env.step(action_step.action)
    train_samples = tf.concat(observations,axis=0)
    train_actions = tf.concat(actions,axis=0)
    return train_samples.numpy(), train_actions.numpy()


if __name__ == '__main__':

    ### Set up hardware
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())    
    # print(tf.config.experimental.list_physical_devices())
    # use_device = '/device:GPU:0'
    
    ### Parameters
    root_dir = r'E:\VladGoogleDrive\Qulab\GKP\sims\Benchmarking_HybridMarkovian4Rounds\supervised_linear'
    all_models_dir = 'all_models'
    filename = 'linear.hdf5'
    logname = 'log_linear.log'
    episodes = 100
    steps = 400
    horizon = 16
    
    ### Environment, policy, and training data
    env = GKP(H=horizon, 
              episode_length=steps,
              batch_size=episodes,
              init='random')
    
    policy = plc.MarkovianPolicyV2(env.time_step_spec())

    train_samples, train_actions = create_train_data(env, policy)
    
    
    ### Callbacks
    ModelCheckpoint_best = ModelCheckpoint(
        filepath=os.path.join(root_dir,filename), 
        monitor='mse', save_best_only=True, 
        mode='auto', period=100)

    path_all = os.path.join(root_dir,all_models_dir)
    ModelCheckpoint_all = ModelCheckpoint(
        filepath=os.path.join(path_all,'{epoch:02d}.hdf5'), 
        monitor='mse', save_best_only=False, 
        mode='auto', period=100)

    # LearningRateScheduler_ = LearningRateScheduler(
    #     lambda t: 1e-2/(1+0.01*t) )
    
    LearningRateScheduler_ = LearningRateScheduler(
        lambda t: 1e-4/(1+0.01*t) )
    
    
    CSVLogger_ = CSVLogger(filename = os.path.join(root_dir,logname), 
                           separator=',', append=False)
    
    PlotLosses_ = PlotLosses()


    ### Model
    # model = Sequential([LSTM(6, input_shape=(horizon,6)),
    #                     Dense(5)]) #activation= 'tanh'

    # model = Sequential([Flatten(input_shape=(horizon,6)),
    #                     Dense(200, activation='relu'),
    #                     Dense(200, activation='relu'),
    #                     Dense(50, activation='relu'),
    #                     Dense(5)])

    model = Sequential([Flatten(input_shape=(horizon,6)),
                        Dense(200, activation='relu'),
                        Dense(5)])

    
    model.compile(optimizer = Adam(),
                  loss = 'mean_squared_error',
                  metrics=['mse'])

    ### Training
    t_start = time()
    history = model.fit(x = train_samples, y = train_actions, 
                        batch_size = 1000, 
                        epochs = 10000, 
                        verbose = 2,
                        callbacks = [ModelCheckpoint_best, 
                                     ModelCheckpoint_all,
                                     LearningRateScheduler_,
                                     CSVLogger_, PlotLosses_])
    t_stop = time()
    
    ### Plotting
    loss = history.history['loss']
    epochs = range(len(loss))
    fig, ax = plt.subplots(1,1)
    ax.plot(epochs, loss, color='red')
    ax.set_title('Loss (Mean Squared Error)')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ### Saving
    # model.save(model_file)
    