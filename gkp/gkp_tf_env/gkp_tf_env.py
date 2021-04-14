# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:50:22 2020

@author: Vladimir Sivak
"""
from abc import ABCMeta, abstractmethod
from math import sqrt, pi

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import complex64 as c64
from tf_agents import specs
from tf_agents.environments import tf_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import tensor_spec
from simulator.utils import measurement, expectation, normalize
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
        reward_kwargs={'reward_mode' : 'zero'},
        encoding='square',
        phase_space_rep='wigner',
        **kwargs):
        """
        Args:
            H (int, optional): Horizon for history returned in observations. Defaults to 1.
            T (int, optional): Periodicity of the 'clock' observation. Defaults to 4.
            attn_step (int, optional): step size for hard-coded attention to
                measurement outcomes. For example, set to 4 to return history 
                of measurement oucomes separated by 4 steps -- when the same 
                stabilizer is measured in the square code. In hexagonal code 
                this can be 2. Defaults to 1.
            episode_length (int, optional): Number of iterations in training episode. Defaults to 20.
            batch_size (int, optional): Vectorized minibatch size. Defaults to 50.
            init (str, optional): Initial quantum state of system. Defaults to "vac".
            reward_kwargs (dict, optional): optional dictionary of parameters 
                for the reward function of RL agent.
            encoding (str, optional): Type of GKP lattice. Defaults to "square".
            phase_space_rep (str, optional): phase space representation to use
                for rendering ('wigner' or 'CF')
            
        """
        # Default simulation parameters
        self.H = H
        self.T = T
        self.attn_step = attn_step
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.init = init
        self.phase_space_rep = phase_space_rep
        
        self.setup_reward(reward_kwargs)
        self.define_stabilizer_code(encoding)
        self._epoch = 1

        # Define action and observation specs
        self.quantum_circuit = self._quantum_circuit
        action_spec = self._quantum_circuit_spec

        observation_spec = {
            'msmt'  : specs.TensorSpec(shape=[self.H], dtype=tf.float32),
            'clock' : specs.TensorSpec(shape=[self.T], dtype=tf.float32),
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
            self.history[a].append(action[a])
        self.history['msmt'].append(obs)
        
        # Make observations of 'msmt' of horizon H, shape=[batch_size,H]
        # measurements are selected with hard-coded attention step.
        # Also add clock of period 'T' to observations, shape=[batch_size,T]
        observation = {}
        H = [self.history['msmt'][-self.attn_step*i-1] for i in range(self.H)]
        H.reverse() # to keep chronological order of measurements
        observation['msmt'] = tf.concat(H, axis=1)
        C = tf.one_hot([self._elapsed_steps%self.T]*self.batch_size, self.T)
        observation['clock'] = C
        observation['const'] = tf.ones(shape=[self.batch_size,1])

        reward = self.calculate_reward(action)
        self._episode_return += reward
        
        if self._episode_ended:
            self._epoch += 1
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
        self.info = {} # use to cache some intermediate results
        # Create initial state
        if self.init in ['vac','X+','X-','Y+','Y-','Z+','Z-']:
            psi = self.states[self.init]
            psi_batch = tf.stack([psi]*self.batch_size)
            self._state = self.info['psi_cached'] = psi_batch
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

        # Initialize history of horizon H with actions=0 and measurements=1 
        self.history = tensor_spec.zero_spec_nest(
            self.action_spec(), outer_dims=(self.batch_size,))
        for key in self.history.keys():
            self.history[key] = [self.history[key]]*self.H
        # Initialize history of horizon H*attn_step with measurements=1 
        m = tf.ones(shape=[self.batch_size,1])
        self.history['msmt'] = [m]*self.H*self.attn_step
        
        # Make observation of horizon H
        observation = {
            'msmt'  : tf.concat(self.history['msmt'][-self.H:], axis=1), 
            'clock' : tf.one_hot([0]*self.batch_size, self.T),
            'const'   : tf.ones(shape=[self.batch_size,1])}
        
        self._current_time_step_ = ts.restart(observation, self.batch_size)
        return self.current_time_step()


    def render(self):
        """
        Render environment to the screen (plot symmetric characteristic function 
        of the cached state).
        
        """
        state = self.info['psi_cached']
        
        # Generate a grid of phase space points
        lim, pts = 4, 81
        x = np.linspace(-lim, lim, pts)
        y = np.linspace(-lim, lim, pts)

        x = tf.constant(x, dtype=c64)
        y = tf.constant(y, dtype=c64)
        
        one = tf.constant([1]*len(y), dtype=c64)
        onej = tf.constant([1j]*len(x), dtype=c64)
        
        grid = tf.tensordot(x, one, axes=0) + tf.tensordot(onej, y, axes=0)
        grid_flat = tf.reshape(grid, [-1])
        
        # Calculate and plot the phase space representation
        if self.phase_space_rep == 'wigner':
            state = tf.broadcast_to(state, [grid_flat.shape[0], state.shape[1]])
            state_translated = tf.linalg.matvec(self.translate(-grid_flat), state)
            W = 1/pi * expectation(state_translated, self.parity, reduce_batch=False)
            W_grid = tf.reshape(W, grid.shape)
    
            fig, ax = plt.subplots(1,1)
            fig.suptitle('Wigner, step %d' %self._elapsed_steps)
            ax.pcolormesh(x, y, np.transpose(W_grid.numpy().real), 
                               cmap='RdBu_r', vmin=-1/pi, vmax=1/pi)
            ax.set_aspect('equal')
        
        if self.phase_space_rep == 'CF':
            C = expectation(state, self.translate(grid_flat), reduce_batch=False)
            C_grid = tf.reshape(C, grid.shape)
            
            fig, axes = plt.subplots(1,2, sharey=True)
            fig.suptitle('step %d' %self._elapsed_steps)
            axes[0].pcolormesh(x, y, np.transpose(C_grid.numpy().real), 
                                cmap='RdBu_r', vmin=-1, vmax=1)
            axes[1].pcolormesh(x, y, np.transpose(C_grid.numpy().imag), 
                                cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0].set_title('Re')
            axes[1].set_title('Im')
            axes[0].set_aspect('equal')
            axes[1].set_aspect('equal')


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

    def define_stabilizer_code(self, encoding):
        """
        Create stabilizer tensors, logical Pauli tensors and GKP state tensors.
        The simulation Hilbert space consists of N levels of the oscillator 
        and, if the 'tensorstate' flag is set, it also includes the qubit. In 
        the latter case the qubit comes first in the tensor product. 
        
        Input:
            encoding (str): Type of GKP lattice (square or hexagonal)
            
        """
        # S is symplectic 2x2 matrix that defines the code subspace
        if encoding == 'square':
            S = np.array([[1, 0], [0, 1]])
        elif encoding == 'hexagonal':
            S = np.array([[1, 1/2], [0, sqrt(3)/2]])*sqrt(2/sqrt(3))
            
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

    def setup_reward(self, reward_kwargs):
        """Setup the reward function based on reward_kwargs. """
        try:
            mode = reward_kwargs.pop('reward_mode')
            assert mode in ['zero',
                            'measurement',
                            'stabilizers',
                            'stabilizers_v2',
                            'pauli', 
                            'fidelity',
                            'overlap',
                            'Fock',
                            'tomography',
                            'remote']
            self.reward_mode = mode
        except: 
            raise ValueError('reward_mode not specified or not supported.') 
        
        if mode == 'zero':
            """
            Required reward_kwargs:
                reward_mode (str): 'zero'
                
            """
            self.calculate_reward = self.reward_zero

        if mode == 'measurement':
            """
            Required reward_kwargs:
                reward_mode (str): 'measurement'
                sample (bool): flag to sample binary outcomes
                
            """
            sample = reward_kwargs['sample']
            self.calculate_reward = \
                lambda args: self.reward_measurement(sample, args)

        if mode == 'pauli':
            """
            Required reward_kwargs:
                reward_mode (str): 'pauli'
                code_flips (bool): flag to count logical code flips
                
            """
            assert 'code_flips' in reward_kwargs.keys()
            if self.init == 'vac':
                raise Exception('Pauli reward not supported for vac')
            code_flips = reward_kwargs['code_flips']
            if code_flips: self.flips = {'X' : 0, 'Z' : 0, 'Y' : 0}
            self.calculate_reward = \
                lambda args : self.reward_pauli(code_flips, args)

        if mode == 'fidelity':
            """
            Required reward_kwargs:
                reward_mode (str): 'fidelity'
                code_flips (bool): flag to count logical code flips
                
            """
            assert 'code_flips' in reward_kwargs.keys()
            if self.init == 'vac':
                raise Exception('Fidelity reward not supported for vac')
            code_flips = reward_kwargs['code_flips']
            if code_flips: self.flips = {'X' : 0, 'Z' : 0, 'Y' : 0}
            self.calculate_reward = \
                lambda args : self.reward_fidelity(code_flips, args)
        
        if mode == 'overlap':
            """
            Required reward_kwargs:
                reward_mode (str): 'overlap'
                target_state (Qobj, type=ket): Qutip object
                postselect_0 (bool): flag to project qubit onto |0>. If False,
                    will project randomly with a sigam_z measurement
                
            """
            assert 'target_state' in reward_kwargs.keys()
            assert 'postselect_0' in reward_kwargs.keys()
            target_projector = qt.ket2dm(reward_kwargs['target_state'])
            target_projector = tf.constant(target_projector.full(), dtype=c64)
            postselect_0 = reward_kwargs['postselect_0']
            self.calculate_reward = \
                lambda args: self.reward_overlap(target_projector, postselect_0)

        if mode == 'Fock':
            """
            Required reward_kwargs:
                reward_mode (str): 'Fock'
                target_state (Qobj, type=ket): Qutip object
                
            """
            assert 'target_state' in reward_kwargs.keys()
            target_projector = qt.ket2dm(reward_kwargs['target_state'])
            target_projector = tf.constant(target_projector.full(), dtype=c64)
            self.calculate_reward = \
                lambda args: self.reward_Fock(target_projector, args)        

        if mode == 'tomography':
            """
            Required reward_kwargs:
                reward_mode (str): 'tomography'
                tomography (str): either 'wigner' or 'CF'
                target_state (Qobj, type=ket): Qutip object
                window_size (float): size of phase space window
                N_alpha (int): number of phase space points to sample
                N_msmt (int): number of measurements per phase space point
                sampling_type (str): either 'abs' or 'square'
            """
            assert reward_kwargs['sampling_type'] in ['abs', 'sqr']
            assert reward_kwargs['tomography'] in ['wigner', 'CF']
            
            target_state = reward_kwargs.pop('target_state')
            target_state = tf.constant(target_state.full(), dtype=c64)
            target_state = tf.transpose(target_state)
            reward_kwargs['target_state'] = target_state
            
            self.calculate_reward = lambda x: self.reward_tomography(**reward_kwargs)

        if mode == 'stabilizers':
            """
            Required reward_kwargs:
                reward_mode (str): 'stabilizers'
                stabilizer_translations (list, float): translation amplitudes
                
            """
            assert 'stabilizer_translations' in reward_kwargs.keys()
            stabilizer_translations = reward_kwargs['stabilizer_translations']
            self.calculate_reward = \
                lambda args: self.reward_stabilizers(stabilizer_translations, args)

        if mode == 'stabilizers_v2':
            """
            Required reward_kwargs:
                reward_mode (str): 'stabilizers_v2'
                Delta (float): envelope trimming amplitude
                beta (float): stabilizer displacement amplitude
                sample (bool): flag to sample measurements or use expectations
                
            """
            for k in ['Delta', 'beta', 'sample']:
                assert k in reward_kwargs.keys()
            sample = reward_kwargs['sample'] 
            self.calculate_reward = lambda args: self.reward_stabilizers_v2(
                reward_kwargs['beta'], reward_kwargs['Delta'], sample)

        if mode == 'remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'remote'
                tomography (str): either 'wigner' or 'CF'
                target_state (Qobj, type=ket): Qutip state-vector object
                window_size (float): size of (symmetric) phase space window
                server_socket (Socket): socket for communication
                amplitude_type (str): either 'displacement' or 'translation'
                epoch_type (str): either 'training' or 'evaluation'
                N_alpha (int): number of phase space points to sample
                N_msmt (int): number of measurements per phase space point
                sampling_type (str): either 'abs' or 'square'
            """
            assert reward_kwargs['tomography'] in ['wigner', 'CF']
            assert reward_kwargs['amplitude_type'] in ['displacement', 'translation']
            assert reward_kwargs['epoch_type'] in ['training', 'evaluation']
            assert reward_kwargs['sampling_type'] in ['abs', 'sqr']
            self.server_socket = reward_kwargs.pop('server_socket')
            
            target_state = reward_kwargs.pop('target_state')
            target_state = tf.constant(target_state.full(), dtype=c64)
            target_state = tf.transpose(target_state)
            reward_kwargs['target_state'] = target_state
            
            self.calculate_reward = lambda x: self.reward_remote(**reward_kwargs)


    def reward_Fock(self, target_projector, *args):
        """
        Reward only on last time step using the measurement of a given Fock 
        state of the oscillator.

        """
        if self._elapsed_steps < self.episode_length:
            z = tf.zeros(self.batch_size, dtype=tf.float32)
        else:            
            if self.tensorstate:
                # measure qubit to disentangle from oscillator
                psi, _ = measurement(self._state, self.P, sample=True)
            else:
                psi = self._state
            overlap = expectation(psi, target_projector, reduce_batch=False)
            z = tf.reshape(tf.math.real(overlap), shape=[self.batch_size])

            obs = tfp.distributions.Bernoulli(probs=z).sample()
            obs = 2*obs -1 # convert to {-1,1}
            z = tf.cast(obs, dtype=tf.float32)
        return z
    

    def reward_overlap(self, target_projector, postselect_0):
        """
        Reward only on last time step using the overlap of the cached state 
        with the target state. The cached state is measured prior to computing
        the overlap, to make sure that the oscillator and qubit are disentangled.
        The agent can learn to arrange things so that this measurement doesn't
        matter (i.e. if they are already disentangled before the measurement). 

        """
        if self._elapsed_steps < self.episode_length:
            z = tf.zeros(self.batch_size, dtype=tf.float32)
        else:            
            if self.tensorstate:
                if postselect_0: # project qubit to |0>
                    psi = tf.linalg.matvec(self.P[0], self._state)
                    psi, _ = normalize(psi)
                else: # randomly project qubit with a measurement
                    psi, _ = measurement(self._state, self.P, sample=True)
            else:
                psi = self._state
            overlap = expectation(psi, target_projector, reduce_batch=False)
            z = tf.reshape(tf.math.real(overlap), shape=[self.batch_size])
            self.info['psi_cached'] = psi
        return z


    def fill_buffer(self, target_state, window_size, tomography, num_samples,
                    sampling_type):
        """
        Fill the buffer of phase space points and corresponding target Wigner
        or characteristic function values. Points are produced with rejection
        sampling proportionally to the |W| or |Re[C]|.
        
        Returns:
            target_state (Tensor([1,N]), c64): tf target state
            window_size (float): size of (symmetric) phase space window
            tomography (str): either 'wigner' or 'CF'
            num_samples (int): number of samples to generate
            sampling_type (str): either 'abs' or 'sqr'
        """
        # Distributions for rejection sampling
        L = window_size/2
        P = tfp.distributions.Uniform(low=[[-L,-L]]*samples, high=[[L,L]]*samples)
        P_v = tfp.distributions.Uniform(low=[0.0]*samples, high=[1.0]*samples)
            
        cond = True
        # mask for asynchronous interruption of rejection sampling
        accepted = tf.zeros([samples], tf.bool) 
        accepted_points = tf.zeros([samples], tf.complex64)
        accepted_targets = tf.zeros([samples], tf.float32)
        # batch rejection sampling of phase space points
        while cond:
            points = tf.squeeze(hf.vec_to_complex(P.sample()))
            if tomography == 'wigner':
                target_state_translated = tf.linalg.matvec(
                    self.translate(-points), target_state)
                W_target = expectation(target_state_translated, 
                                       self.parity, reduce_batch=False)
                target = tf.math.real(tf.squeeze(W_target))
            if tomography == 'CF':
                C_target = expectation(target_state, self.translate(-points), 
                                       reduce_batch=False)
                target = tf.math.real(tf.squeeze(C_target))
            if sampling_type == 'abs': 
                V = tf.math.abs(target)
            elif sampling_type == 'sqr': 
                V = tf.math.abs(target)**2
            reject = P_v.sample() > V
            # mask which streams should be interrupted at this iteration
            mask = tf.logical_and(tf.logical_not(reject), tf.logical_not(accepted))
            accepted = tf.logical_or(mask, accepted)
            accepted_points = tf.where(mask, points, accepted_points)
            accepted_targets = tf.where(mask, target, accepted_targets)
            cond = not tf.reduce_all(accepted)
        
        buffer = list(accepted_points)
        target_vals = list(accepted_targets)
        return buffer, target_vals

    def collect_tomography_measurements(self, tomography, mini_buffer, N_msmt):
        """
        Given the buffer of 'N_alpha' phase space points, do 'N_msmt'
        tomography measurements per point. 
        
        Args:
            tomography (str): type of tomography, either 'Wigner' or 'CF'
            mini_buffer (list(c64)): a list of phase space points
            N_msmt (int): number of measurements to do per phase space point
        
        Returns:
            msmt (Tensor([2, batch_size, N_alpha, N_msmt], float32)): tensor 
                of measurement outomes; msmt[0] is the first measurement in 
                the tomography reward circuit, which disentangles the qubit
                and oscillator, msmt[1] is the actual tomography measurement. 
        """
        N_alpha = len(mini_buffer)
        M1_buffer, M2_buffer = [], []
        
        for i in range(N_alpha):
            # take 1 point from the buffer and replicate it for the batch
            points = tf.broadcast_to(mini_buffer[i], [self.batch_size])
            M1, M2 = [], []

            for j in range(N_msmt):
                # first measure the qubit to disentangle from oscillator
                psi = self.info['psi_cached']
                if self.tensorstate:
                    psi, m1 = measurement(psi, self.P, sample=True)
                else:
                    m1 = tf.ones([self.batch_size])
                M1.append(tf.squeeze(m1))
                
                # do 1 tomography measurement in this phase space point
                if tomography == 'wigner':
                    translations = self.translate(-points)
                    psi = tf.linalg.matvec(translations, psi)
                    _, m2 = self.phase_estimation(psi, self.parity,
                                angle=tf.zeros(self.batch_size), sample=True)
                
                if tomography == 'CF':
                    translations = self.translate(points)
                    _, m2 = self.phase_estimation(psi, translations,
                                angle=tf.zeros(self.batch_size), sample=True)
                M2.append(tf.squeeze(m2))
            
            M1_buffer.append(M1)
            M2_buffer.append(M2)
        msmt = tf.cast([M1_buffer, M2_buffer], tf.float32)
        msmt = tf.transpose(msmt, perm=[0,3,1,2])
        return msmt
    

    def reward_tomography(self, target_state, window_size, tomography, 
                          N_alpha, N_msmt, sampling_type):
        """
        Reward only on last time step using the empirical approximation of the 
        overlap of the prepared state with the target state. This overlap is
        computed in characteristic-function-tomography-like fashing.
        
        The state is measured prior to acquiring a tomography point on the 
        oscillator to make sure that the oscillator and qubit are disentangled.
        The agent can learn to arrange things so that this measurement doesn't
        matter (i.e. if they are already disentangled before the measurement). 

        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps < self.episode_length:
            return tf.zeros(self.batch_size, dtype=tf.float32)

        def penalty_coeff_schedule(n):
            return 1

        penalty_coeff = penalty_coeff_schedule(self._epoch)
        
        # populate a new mini-buffer for each epoch
        mini_buffer, target_vals = self.fill_buffer(
                target_state, window_size, tomography, N_alpha, sampling_type)
        
        # get array of outcomes of shape [2, batch_size, N_alpha, N_msmt]
        msmt = self.collect_tomography_measurements(
            tomography, mini_buffer, N_msmt)
        
        # create mask based on the first "disentangling" measurement
        # construct fidelity estimator and penalty term
        mask = tf.where(msmt[0]==1, 1.0, 0.0) # [batch_size, N_alpha, N_msmt]
        targets = tf.reshape(target_vals, [1, len(target_vals), 1])
        
        if sampling_type == 'abs':
            Z = msmt[1] * tf.math.sign(targets) * mask
        elif sampling_type == 'sqr'
            Z = msmt[1] / targets * mask
        
        Z = Z - penalty_coeff * (1-mask)
        z = tf.math.reduce_mean(Z, axis=[1,2])
        return z

    def reward_remote(self, target_state, window_size, tomography,
                      amplitude_type, epoch_type, N_alpha, N_msmt,
                      sampling_type):
        """
        Send the action sequence to remote environment and receive rewards.
        The data received from the remote env should be sigma_z measurement 
        outcomes of shape [2, batch_size, N_alpha, N_msmt] where the first
        measurement is disentangling and second measurement is tomography. 
        
        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps != self.episode_length:
            return tf.zeros(self.batch_size, dtype=tf.float32)

        penalty_coeff = 1.0
        
        action_batch = {}
        for a in self.history.keys() - ['msmt']:
            # reshape to [batch_size, T, action_dim] 
            action_batch[a] = np.transpose(self.history[a][1:], axes=[1,0,2])
 
        # populate a new mini-buffer for each epoch
        mini_buffer, targets = self.fill_buffer(
                target_state, window_size, tomography, N_alpha, sampling_type)

        np_targets = [C.numpy() for C in targets]
        np_mini_buffer = [alpha.numpy() for alpha in mini_buffer]
        
        if amplitude_type == 'displacement':
            np_mini_buffer = [alpha / sqrt(2) for alpha in np_mini_buffer]
        
        # send action sequence and phase space points to remote client
        message = dict(action_batch=action_batch, 
                       mini_buffer=np_mini_buffer,
                       targets=np_targets,
                       batch_size=self.batch_size,
                       N_alpha=N_alpha, N_msmt=N_msmt,
                       epoch_type=epoch_type,
                       epoch=self._epoch)
        
        self.server_socket.send_data(message)
        
        # receive array of outcomes of shape [2, batch_size, N_alpha, N_msmt]
        msmt, done = self.server_socket.recv_data()
        
        if epoch_type == 'evaluation':
            return tf.zeros(self.batch_size, dtype=tf.float32)
        
        msmt = tf.cast(msmt, tf.float32)
        mask = tf.where(msmt[0]==1, 1.0, 0.0) # [batch_size, N_alpha, N_msmt]
        targets = tf.reshape(targets, [1, len(targets), 1])
        
        if sampling_type == 'abs':
            Z = msmt[1] * tf.math.sign(targets) * mask
        elif sampling_type == 'sqr'
            Z = msmt[1] / targets * mask
        
        Z = Z - penalty_coeff * (1-mask)
        z = tf.math.reduce_mean(Z, axis=[1,2])
        return z
        

    @tf.function
    def reward_zero(self, *args):
        """
        Reward is always zero (use when not training).
        
        """
        return tf.zeros(self.batch_size, dtype=tf.float32)


    def reward_measurement(self, sample, *args):
        """
        Reward is simply the outcome of sigma_z measurement.
        
        """
        psi, z = measurement(self.info['psi_cached'], self.P, sample)
        return tf.squeeze(z)


    def reward_pauli(self, code_flips, act):
        """
        Reward only on last time step with the result of measurement of logical
        Pauli operator using cached wavefunction (after feedback translation).
        Such reward lets the agent directly optimize T1.

        Input:
            act -- actions at this time step; shape=(batch_size,act_dim)
            code_flips (bool): flag to control code flips count "in software"
            
        """
        # Count code flips that affect cached state
        if code_flips and 'alpha' in act.keys():
            self.count_code_flips(act, 'alpha')
            
        if self._elapsed_steps < self.episode_length:
            z = tf.zeros(self.batch_size, dtype=tf.float32)
            if code_flips: self.count_code_flips(act, 'beta')
        else:
            pauli = [self.code_map[self._original[i][0]]
                         for i in range(self.batch_size)]
            pauli = tf.convert_to_tensor(pauli, dtype=c64)
            phi = tf.zeros(self.batch_size)
            pauli = self.translate(pauli)
            _, z = self.phase_estimation(self.info['psi_cached'], pauli, 
                                         angle=phi, sample=True)
            z = tf.cast(z, dtype=tf.float32)
            z = tf.reshape(z, shape=(self.batch_size,))
            # Correct for code flips
            if code_flips: 
                z *= self.undo_code_flips()
                self.flips = {'X' : 0, 'Z' : 0, 'Y' : 0}
        return z


    def reward_stabilizers(self, stabilizer_translations, *args):
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
            stabilizers = [stabilizer_translations[int(m)] for m in mask]
            stabilizers = tf.convert_to_tensor(stabilizers, dtype=c64)
            phi = tf.zeros(self.batch_size)
            stabilizers = self.translate(stabilizers)
            
            if self.tensorstate:
                psi, m = measurement(self._state, self.P, sample=True)
                mask = tf.squeeze(tf.where(m==1, 1.0, 0.0))
            else:
                psi = self.info['psi_cached']
                mask = tf.ones([self.batch_size])

            _, z = self.phase_estimation(psi, stabilizers, 
                                         angle=phi, sample=True)
            z = tf.cast(z, dtype=tf.float32)
            z = tf.reshape(z, shape=(self.batch_size,)) * mask
        return z


    def reward_stabilizers_v2(self, beta, Delta, sample):
        """
        Use finite-energy stabilizers instead of ideal stabilizers. These have 
        a parameter 'Delta' controlling the squeezing level and 'beta'
        controlling the stabilizer magnitude (the direction is randomly
        sampled to be e^{0j} or e^{1j*pi/2}).
        
        See this paper by Baptiste Royer for more details:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260509
            
        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps < self.episode_length:
            return tf.zeros(self.batch_size, dtype=tf.float32)
        
        # sample random stabilizer directions (+/-x, +/-p)
        mask = tf.random.uniform([self.batch_size])
        theta_x = tf.where(mask>3/4, 0., pi)*tf.where(mask>1/2, 1., 0.)
        theta_p = tf.where(mask>1/4, pi/2, -pi/2)*tf.where(mask<1/2, 1., 0.)
        theta = theta_x + theta_p
        theta = tf.cast(theta, c64)
        
        if self.tensorstate:
            raise ValueError('Only <Oscillator> Hilbert space is supported.')
        
        eps = beta * Delta**2
        # Create Kraus operators corresponding to finite-energy stabilizers
        D_e = self.displace((eps+0j)*tf.math.exp(1j*theta)/2)
        D_b = self.displace(-1j*beta*tf.math.exp(1j*theta)/2)
        
        chunk1 = tf.linalg.matmul(D_b, D_e - 1j*tf.linalg.adjoint(D_e))
        chunk2 = tf.linalg.matmul(tf.linalg.adjoint(D_b), \
                                  -1j*D_e + tf.linalg.adjoint(D_e))
        Kraus = {}
        Kraus[0] = 1/(2*sqrt(2))*(chunk2 + chunk1)
        Kraus[1] = 1/(2*sqrt(2))*(chunk2 - chunk1)

        psi, z = measurement(self._state, Kraus, sample=sample)
        return tf.squeeze(z)


    

    def reward_fidelity(self, code_flips, act):
        """
        Reward only on last time step with the result of measurement of logical
        Pauli operator using cached wavefunction (after feedback translation).
        Such reward lets the agent directly optimize T1.

        Input:
            act -- actions at this time step; shape=(batch_size,act_dim)
            code_flips (bool): flag to control code flips count "in software"

        """
        # Count code flips that affect cached state
        if code_flips and 'alpha' in act.keys():
            self.count_code_flips(act, 'alpha')

        # Measure the Pauli expectation on cached state 
        pauli = [self.code_map[self._original[i][0]]
                     for i in range(self.batch_size)]
        pauli = tf.convert_to_tensor(pauli, dtype=c64)
        phi = tf.zeros(self.batch_size)
        pauli = self.translate(pauli)
        _, z = self.phase_estimation(self.info['psi_cached'], pauli, 
                                     angle=phi, sample=False)
        z = tf.cast(z, dtype=tf.float32)
        z = tf.reshape(z, shape=(self.batch_size,))
        
        # Correct the Pauli expectation if the code flipped
        if code_flips: z *= self.undo_code_flips()
        # Count code flips that happened after the state was cached
        if code_flips: self.count_code_flips(act, 'beta')
        
        if self._elapsed_steps == self.episode_length and code_flips:
            self.flips = {'X' : 0, 'Z' : 0, 'Y' : 0}
        
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
            atol = abs(ref) / 2
            self.flips[a] += tf.where(tf.math.abs(amp-ref) < atol, 1, 0)
            self.flips[a] += tf.where(tf.math.abs(amp+ref) < atol, 1, 0)
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
