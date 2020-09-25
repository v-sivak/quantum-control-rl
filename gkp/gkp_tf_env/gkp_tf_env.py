# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:50:22 2020

@author: Vladimir Sivak
"""
from abc import ABCMeta, abstractmethod
from math import sqrt, pi

import numpy as np
import qutip as qt
import tensorflow as tf
from tensorflow import complex64 as c64
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec

from gkp.gkp_tf_env import helper_functions as hf


class GKP(tf_environment.TFEnvironment, metaclass=ABCMeta):
    """
    Custom environment that follows TensorFlow Agents interface and allows to
    train a reinforcement learning agent to find quantum control policies.

    This implementation heavily relies on TensorFlow to do fast computations
    in parallel on GPU by adding batch dimension to all tensors. The speedup
    over all-Qutip implementation is about x100 on NVIDIA RTX 2080Ti.

    This is the base environment class for quantum control problems which 
    incorporates simulation-independet methods. The QuantumCircuit subclasses 
    inherit from this base class and from a simulation class. Subclasses
    implementat 'quantum_circuit' which is ran at each time step. RL agent's 
    actions are parametrized according to the sequence of gates applied at
    each time step, as defined by 'quantum_circuit'.

    Environment step() method returns TimeStep tuple whose 'observation'
    attribute stores the finite-horizon history of applied actions, measurement
    outcomes and state wavefunctions. User needs to define a wrapper for the
    environment if some components of this observation are to be discarded.
    """
    def __init__(
        self,
        *args,
        # Optional kwargs
        H=1,
        T=4,
        attn_step=1,
        episode_length=20,
        batch_size=50,
        init="vac",
        reward_mode="zero",
        encoding="square",
        stabilizer_translations=None,
        **kwargs
    ):
        """
        Args:
            H (int, optional): Horizon for history returned in observations. Defaults to 1.
            T (int, optional): Periodicity of the 'clock' observation. Defaults to 4.
            attn_step (int, optional): Step for attention to measurement outcomes. Defaults to 1.
            episode_length (int, optional): Number of iterations in training episode. Defaults to 20.
            batch_size (int, optional): Vectorized minibatch size. Defaults to 50.
            init (str, optional): Initial quantum state of system. Defaults to "vac".
            reward_mode (str, optional): Type of reward for RL agent. Defaults to "zero".
            encoding (str, optional): Type of GKP lattice. Defaults to "square".
            stabilizer_translations (list, optional): list of stabilizer translation amplitudes.
            
        """
        # Default simulation parameters
        self.H = H
        self.T = T
        self.attn_step = attn_step
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.init = init
        self.reward_mode = reward_mode
        self.stabilizer_translations = stabilizer_translations

        if encoding == 'square':
            S = np.array([[1, 0], [0, 1]])
        elif encoding == 'hexagonal':
            S = np.array([[1, 1/2], [0, sqrt(3)/2]])*sqrt(2/sqrt(3))
        self.define_stabilizer_code(S)

        # Define action and observation specs
        self.quantum_circuit = self._quantum_circuit
        action_spec = self._quantum_circuit_spec

        observation_spec = {
            'msmt'  : specs.TensorSpec(shape=[self.H,1], dtype=tf.float32),
            'clock' : specs.TensorSpec(shape=[1,self.T], dtype=tf.float32),
            'const' : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        time_step_spec = ts.time_step_spec(observation_spec)

        super(GKP, self).__init__(time_step_spec, action_spec, self.batch_size)
        

    ### STANDARD


    def _step(self, action):
        """
        Execute one time step in the environment.
        
        Input:
            action -- dictionary of batched actions
        
        Output:
            TimeStep object (see tf-agents docs)  
            
        """
        self._state, info, obs = self.quantum_circuit(self._state, action)
        self.info['psi_cached'] = info
        # Calculate rewards
        self._elapsed_steps += 1
        self._episode_ended = (self._elapsed_steps == self.episode_length)
        
        # Add dummy time dimension to tensors and append them to history
        for a in action.keys():
            self.history[a].append(tf.expand_dims(action[a], axis=1))
        self.history['msmt'].append(tf.expand_dims(obs, axis=1))
        
        # Make observations of 'msmt' of horizon H, shape=[batch_size,H,1]
        # measurements are selected with hard-coded attention step.
        # Also add clock of period 'T' to observations, shape=[batch_size,1,T]
        observation = {}
        H = [self.history['msmt'][-self.attn_step*i-1] for i in range(self.H)]
        H.reverse() # to keep chronological order of measurements
        observation['msmt'] = tf.concat(H, axis=1)
        C = tf.one_hot([[self._elapsed_steps%self.T]]*self.batch_size, self.T)
        observation['clock'] = C
        observation['const'] = tf.ones(shape=[self.batch_size,1])

        reward = self.calculate_reward(action)
        self._episode_return += reward
        
        if self._episode_ended:
            self._current_time_step_ = ts.termination(observation, reward)
        else:
            self._current_time_step_ = ts.transition(observation, reward)
        return self.current_time_step()


    def _reset(self):
        """
        Reset the state of the environment to an initial state. States are 
        represented as batched tensors. 
        
        Input:
            init -- type of states to create in a batch
                * 'vac': vacuum state
                * 'X+','X-','Y+','Y-','Z+','Z-': one of the cardinal states
                * 'random': sample batch of random states from 'X+','Y+','Z+'
                
        Output:
            TimeStep object (see tf-agents docs)
            
        """
        # Create initial state
        if self.init in ['vac','X+','X-','Y+','Y-','Z+','Z-']:
            psi = self.states[self.init]
            psi_batch = tf.stack([psi]*self.batch_size)
            self._state = psi_batch
            self._original = np.array([self.init]*self.batch_size)
        elif self.init == 'random':
            self._original = np.random.choice(['X+','Y+','Z+'], 
                                              size=self.batch_size)
            psi_batch = [self.states[init] for init in self._original]
            psi_batch = tf.convert_to_tensor(psi_batch, dtype=c64)
            self._state = psi_batch

        # Bookkeeping of episode progress
        self._episode_ended = False
        self._elapsed_steps = 0
        self._episode_return = 0
        self.info = {} # use to cache some intermediate results

        # Initialize history of horizon H with actions=0 and measurements=1 
        self.history = tensor_spec.zero_spec_nest(
            self.action_spec(), outer_dims=(self.batch_size,))
        for key in self.history.keys():
            self.history[key] = [self.history[key]]*self.H
        # Initialize history of horizon H*attn_step with measurements=1 
        m = tf.ones(shape=[self.batch_size,1,1])
        self.history['msmt'] = [m]*self.H*self.attn_step
        
        
        # Make observation of horizon H, shape=[batch_size,H,dim] of each
        observation = {
            'msmt'  : tf.concat(self.history['msmt'][-self.H:], axis=1), 
            'clock' : tf.zeros(shape=[self.batch_size,1,self.T]),
            'const'   : tf.ones(shape=[self.batch_size,1])}

        # Will keep track of code flips in symmetrized phase estimation
        if self.reward_mode in ['pauli_with_code_flips', 
                                'fidelity_with_code_flips']: 
            self.flips = {'X' : 0, 'Z' : 0, 'Y' : 0}
        
        self._current_time_step_ = ts.restart(observation, self.batch_size)
        return self.current_time_step()    


    def render(self):
        """
        Render environment to the screen (plot Wigner function).
        
        """
        hf.plot_wigner_tf_wrapper(self._state, tensorstate=self.tensorstate)


    def _current_time_step(self):
        return self._current_time_step_
    

    ### GKP - SPECIFIC
    
    
    @abstractmethod
    def _quantum_circuit(self, psi, action):
        """
        Quantum circuit to run on every step, to be defined by the subclass.
        Input:
            psi (Tensor([batch_size,N], c64)): batch of states
            action (dict, 'alpha' : Tensor([batch_size,1,2], tf.float32),
                    'beta': Tensor([batch_size,1,2], tf.float32), etc): 
                    dictionary of actions with keys representing different 
                    operations (translations, rotations etc) as required by 
                    the subclasses. 

        Output:
            psi_final (Tensor([batch_size,N], c64)): batch of final states
            psi_cached (Tensor([batch_size,N], c64)) batch of cached states
            obs (Tensor([batch_size,1], float32)) measurement outcomes; 
                In open-loop control problems can return a tensor of zeros.
        """
        pass

    def define_stabilizer_code(self, S):
        """
        Create stabilizer tensors, logical Pauli tensors and GKP state tensors.
        The simulation Hilbert space consists of N levels of the oscillator 
        and, if the 'tensorstate' flag is set, it also includes the qubit. In 
        the latter case the qubit comes first in the tensor product. 
        
        Input:
            S   -- symplectic 2x2 matrix that defines the code subspace
            
        """
        stabilizers, pauli, states, self.code_map = \
            hf.GKP_state(self.tensorstate, self.N, S)
        # Convert to tensorflow tensors.
        self.stabilizers = {key : tf.constant(val.full(), dtype=c64)
                            for key, val in stabilizers.items()}
        self.pauli = {key : tf.constant(val.full(), dtype=c64)
                      for key, val in pauli.items()}
        self.states = {key : tf.squeeze(tf.constant(val.full(), dtype=c64))
                       for key, val in states.items()}
        vac = qt.basis(2*self.N,0) if self.tensorstate else qt.basis(self.N,0)
        self.states['vac'] = tf.squeeze(tf.constant(vac.full(), dtype=c64))
        

    ### REWARD FUNCTIONS


    @tf.function
    def reward_zero(self, act):
        """
        Reward is always zero (use when not training).
        
        """
        return tf.zeros(self.batch_size, dtype=tf.float32)


    def reward_pauli(self, act, code_flips=True):
        """
        Reward only on last time step with the result of measurement of logical
        Pauli operator using cached wavefunction (after feedback translation).
        Such reward lets the agent directly optimize T1.

        Input:
            act -- actions at this time step; shape=(batch_size,act_dim)
            code_flips (bool): flag to control code flips count "in software"
            
        """
        if code_flips: self.count_code_flips(act, 'alpha')
        
        if self._elapsed_steps < self.episode_length:
            z = tf.zeros(self.batch_size, dtype=tf.float32)
            if code_flips: self.count_code_flips(act, 'beta')
        else:
            pauli = [self.code_map[self._original[i][0]]
                         for i in range(self.batch_size)]
            pauli = tf.convert_to_tensor(pauli, dtype=c64)
            phi = tf.zeros(self.batch_size)
            _, z = self.phase_estimation(self.info['psi_cached'], pauli, 
                                         angle=phi, sample=True)
            z = tf.cast(z, dtype=tf.float32)
            z = tf.reshape(z, shape=(self.batch_size,))
            # Correct for code flips
            if code_flips: z *= self.undo_code_flips()
        return z


    def reward_stabilizers(self, *args):
        """
        Reward only on last time step with the result of measurement of one of
        the stabilizers using cached wavefunction (after feedback translation).
            
        """
        # Calculate reward
        if self._elapsed_steps < self.episode_length:
            z = tf.zeros(self.batch_size, dtype=tf.float32)
        else:
            # TODO: this works only for 2 stabilizers, generalize
            mask = tf.random.categorical(
                tf.math.log([[0.5, 0.5] for _ in range(self.batch_size)]),1)
            stabilizers = [self.stabilizer_translations[int(m)] for m in mask]
            stabilizers = tf.convert_to_tensor(stabilizers, dtype=c64)
            phi = tf.zeros(self.batch_size)
            _, z = self.phase_estimation(self.info['psi_cached'], stabilizers, 
                                         angle=phi, sample=True)
            z = tf.cast(z, dtype=tf.float32)
            z = tf.reshape(z, shape=(self.batch_size,))
        return z
    

    def reward_fidelity(self, act, code_flips):
        """
        Reward only on last time step with the result of measurement of logical
        Pauli operator using cached wavefunction (after feedback translation).
        Such reward lets the agent directly optimize T1.

        Input:
            act -- actions at this time step; shape=(batch_size,act_dim)
            code_flips (bool): flag to control code flips count "in software"

        """
        # Count code flips that affect cached state
        if code_flips: self.count_code_flips(act, 'alpha')

        # Measure the Pauli expectation on cached state 
        pauli = [self.code_map[self._original[i][0]]
                     for i in range(self.batch_size)]
        pauli = tf.convert_to_tensor(pauli, dtype=c64)
        phi = tf.zeros(self.batch_size)
        _, z = self.phase_estimation(self.info['psi_cached'], pauli, 
                                     angle=phi, sample=False)
        z = tf.cast(z, dtype=tf.float32)
        z = tf.reshape(z, shape=(self.batch_size,))
        
        # Correct the Pauli expectation if the code flipped
        if code_flips: z *= self.undo_code_flips()
        # Count code flips that happened after the state was cached
        if code_flips: self.count_code_flips(act, 'beta')
        return z


    def count_code_flips(self, action, key):
        """
        Compare the 'action[key]' amplitude to one of the amplitudes that flip
        the code and increment the corresponding counter of X, Y, or Z flips.
        
        """
        amp = hf.vec_to_complex(action[key])
        ref_amps = {'alpha' : ['X','Y','Z'], 'beta' : ['S_x','S_y','S_z']}
        for a, b in zip(['X','Y','Z'], ref_amps[key]):
            ref = self.code_map[b]
            self.flips[a] += tf.where(amp==ref,1,0) + tf.where(amp==-ref,1,0)
        return 


    def undo_code_flips(self):
        """
        Undo the logical code flips "in software" by counting how many Pauli
        operators were applied. Returns a mask for each trajectory with 1s
        if the code didn't flip and -1s if it flipped.
        
        """
        # For 'X+', undo Z and Y flips
        mask = np.where(self._original == 'X+', 1.0, 0.0)
        flips = tf.math.floormod(self.flips['Z']+self.flips['Y'], 2)
        coeff = tf.where(flips==0, 1.0, -1.0) * mask
        # For 'Z+', undo X and Y flips
        mask = np.where(self._original == 'Z+', 1.0, 0.0)
        flips = tf.math.floormod(self.flips['X']+self.flips['Y'], 2)
        coeff += tf.where(flips==0, 1.0, -1.0) * mask
        # For 'Y+', undo X and Z flips
        mask = np.where(self._original == 'Y+', 1.0, 0.0)
        flips = tf.math.floormod(self.flips['X']+self.flips['Z'], 2)
        coeff += tf.where(flips==0, 1.0, -1.0) * mask
        return coeff


    ### PROPERTIES
    
    
    @property
    def reward_mode(self):
        return self._reward_mode

    @reward_mode.setter
    def reward_mode(self, mode):
        try:
            assert mode in ['zero', 'stabilizers', 'pauli', 'fidelity',
                            'pauli_with_code_flips',
                            'fidelity_with_code_flips']
            self._reward_mode = mode
            if mode == 'zero':
                self.calculate_reward = self.reward_zero
            if mode == 'pauli':
                if self.init == 'vac':
                    raise Exception('Pauli reward not supported for vac')
                self.calculate_reward = lambda a : self.reward_pauli(a, False)
            if mode == 'fidelity':
                if self.init == 'vac':
                    raise Exception('Fidelity reward not supported for vac')
                self.calculate_reward = lambda a : self.reward_fidelity(a, False)
            if mode == 'stabilizers':
                self.calculate_reward = self.reward_stabilizers
            if mode == 'pauli_with_code_flips':
                if self.init == 'vac':
                    raise Exception('Pauli reward not supported for vac')
                self.calculate_reward = lambda a : self.reward_pauli(a, True)
            if mode == 'fidelity_with_code_flips':
                if self.init == 'vac':
                    raise Exception('Fidelity reward not supported for vac')
                self.calculate_reward = lambda a : self.reward_fidelity(a, True)
        except: 
            raise ValueError('Reward mode not supported.') 
    
    @property
    def init(self):
        return self._init
    
    @init.setter
    def init(self, val):
        try:
            assert val in ['vac','random','X+','X-','Y+','Y-','Z+','Z-']
            self._init = val
        except:
            raise ValueError('Initial state not supported.')
    
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        if 'code_map' in self.__dir__():
            raise ValueError('Cannot change batch_size after initialization.')
        try:
            assert size>0 and isinstance(size,int)
            self._batch_size = size
        except:
            raise ValueError('Batch size should be positive integer.')

    @property
    @abstractmethod
    def N(self):
        """
        Size of the oscillator Hilbert space truncation (not including the qubit)
        """
        pass

    @property
    @abstractmethod
    def tensorstate(self):
        """
        Boolean to indicate if the oscillator space is doubled by the qubit or not.
        """
        pass
