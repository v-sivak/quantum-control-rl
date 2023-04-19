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
from simulator.utils import measurement, expectation, normalize, basis, batch_dot
from rl_tools.tf_env import helper_functions as hf

class TFEnvironmentQuantumControl(tf_environment.TFEnvironment, metaclass=ABCMeta):
    """
    Custom environment that follows TensorFlow Agents interface and allows to
    train a reinforcement learning agent to find quantum control policies.

    This implementation heavily relies on TensorFlow to do fast computations
    in parallel on GPU by adding batch dimension to all tensors. The speedup
    over all-Qutip implementation is about x100 on NVIDIA RTX 2080Ti.

    This is the base environment class for quantum control problems which
    incorporates simulation-independet methods. The QuantumCircuit subclasses
    inherit from this base class and from a simulation class. Subclasses
    implement 'control_circuit' which is ran at each time step. RL agent's
    actions are parametrized according to the sequence of gates applied at
    each time step, as defined by 'control_circuit'.

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

        """
        # Default simulation parameters
        self.H = H
        self.T = T
        self.attn_step = attn_step
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.init = init

        self.setup_reward(reward_kwargs)
        self._epoch = 0

        # Define action and observation specs
        self.control_circuit = self._control_circuit
        action_spec = self._control_circuit_spec

        observation_spec = {
            'msmt'  : specs.TensorSpec(shape=[self.H], dtype=tf.float32),
            'clock' : specs.TensorSpec(shape=[self.T], dtype=tf.float32),
            'const' : specs.TensorSpec(shape=[1], dtype=tf.float32)}
        time_step_spec = ts.time_step_spec(observation_spec)

        super().__init__(time_step_spec, action_spec, self.batch_size)


    ### STANDARD


    def _step(self, action):
        """
        Execute one time step in the environment.

        Input:
            action -- dictionary of batched actions

        Output:
            TimeStep object (see tf-agents docs)

        """
        self._state, info, obs = self.control_circuit(self._state, action)
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
        if self.init in ['vac']:
            psi = self.states[self.init]
            psi_batch = tf.stack([psi]*self.batch_size)
            self._state = self.info['psi_cached'] = psi_batch
            self._original = np.array([self.init]*self.batch_size)

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
      pass

    def _current_time_step(self):
        return self._current_time_step_

    @abstractmethod
    def _control_circuit(self, psi, action):
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


    ### REWARD FUNCTIONS

    def setup_reward(self, reward_kwargs):
        """Setup the reward function based on reward_kwargs. """
        try:
            mode = reward_kwargs.pop('reward_mode')
            # mode = reward_kwargs['reward_mode']
            assert mode in ['zero',
                            'measurement',
                            'stabilizers',
                            'stabilizers_v2',
                            'pauli',
                            'fidelity',
                            'overlap',
                            'fock',
                            'tomography',
                            'tomography_remote',
                            'fock_remote',
                            'stabilizer_remote',
                            'gate',
                            'gate_fidelity']
            self.reward_mode = mode
        except:
            raise ValueError('reward_mode not specified or not supported.')

        if mode == 'zero':
            """
            Required reward_kwargs:
                reward_mode (str): 'zero'

            """
            self.calculate_reward = self.reward_zero


        if mode == 'gate':
            """
            Required reward_kwargs:
                reward_mode (str): 'gate'
                gate matrix (Array([2,2], c64)): logical gate matrix, 2x2
                tomography (str): either 'wigner' or 'CF'
                window_size (float): size of phase space window
                N_alpha (int): number of phase space points to sample
                N_msmt (int): number of measurements per phase space point
                sampling_type (str): either 'abs' or 'square'
            """
            matrix = reward_kwargs.pop('gate_matrix')
            gate_map = self.gate_matrix_to_gate_map(matrix)
            assert reward_kwargs['sampling_type'] in ['abs', 'sqr']
            assert reward_kwargs['tomography'] in ['wigner', 'CF']
            reward_kwargs['gate_map'] = gate_map

            self.calculate_reward = lambda x: self.reward_gate(**reward_kwargs)

        if mode == 'gate_fidelity':
            """
            Required reward_kwargs:
                reward_mode (str): 'gate_fidelity'
                gate matrix (Array([2,2], c64)): logical gate matrix, 2x2
            """
            matrix = reward_kwargs.pop('gate_matrix')
            gate_map = self.gate_matrix_to_gate_map(matrix)
            reward_kwargs['gate_map'] = gate_map
            self.calculate_reward = lambda x: self.reward_gate_fidelity(**reward_kwargs)

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

        if mode == 'fock':
            """
            Required reward_kwargs:
                reward_mode (str): 'fock'
                target_state (Qobj, type=ket): Qutip object
                N_msmt (int): number of measurements
                error_prob (float): with this probability the measurement
                    outcome will be replaced with 'g'.

            """
            target_state = reward_kwargs.pop('target_state')
            projector = tf.constant(qt.ket2dm(target_state).full(), dtype=c64)
            reward_kwargs['target_projector'] = projector
            self.calculate_reward = lambda x: self.reward_fock(**reward_kwargs)

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

        if mode == 'tomography_remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'tomography_remote'
                tomography (str): either 'wigner' or 'CF'
                target_state (qt.Qobj or tf.Tensor): state-vector object
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
            if isinstance(target_state, qt.Qobj):
                target_state = tf.constant(target_state.full(), dtype=c64)
                target_state = tf.transpose(target_state)
            elif not isinstance(target_state, tf.Tensor):
                raise NotImplementedError
            reward_kwargs['target_state'] = target_state

            self.calculate_reward = \
                lambda x: self.reward_tomography_remote(**reward_kwargs)

        if mode == 'fock_remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'fock_remote'
                fock (int): photon number
                N_msmt (int): number of measurements per protocol
                epoch_type (str): either 'training' or 'evaluation'
                server_socket (Socket): socket for communication
            """
            self.server_socket = reward_kwargs.pop('server_socket')
            self.calculate_reward = \
                lambda x: self.reward_fock_remote(**reward_kwargs)

        if mode == 'stabilizer_remote':
            """
            Required reward_kwargs:
                reward_mode (str): 'stabilizer_remote'
                N_msmt (int): number of measurements per protocol
                epoch_type (str): either 'training' or 'evaluation'
                stabilizer_amplitudes (list, float): list of stabilizer dis-
                    placement amplitudes.
                stabilizer_signs (list, flaot): list of stabilizer signs
                server_socket (Socket): socket for communication
            """
            self.server_socket = reward_kwargs.pop('server_socket')
            self.calculate_reward = \
                lambda x: self.reward_stabilizer_remote(**reward_kwargs)


    def reward_fock_remote(self, fock, N_msmt, epoch_type):
        """
        Send the action sequence to remote environment and receive rewards.
        The data received from the remote env should be sigma_z measurement
        outcomes of shape [2, batch_size, N_msmt] where the first measurement
        is disentangling and second measurement is target fock projector.

        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps != self.episode_length:
            return tf.zeros(self.batch_size, dtype=tf.float32)

        penalty_coeff = 1.0

        action_batch = {}
        for a in self.history.keys() - ['msmt']:
            # reshape to [batch_size, T, action_dim]
            action_batch[a] = np.transpose(self.history[a][1:], axes=[1,0,2])

        # send action sequence and phase space points to remote client
        message = dict(action_batch=action_batch,
                       batch_size=self.batch_size,
                       N_msmt=N_msmt,
                       fock=fock,
                       epoch_type=epoch_type,
                       epoch=self._epoch)

        self.server_socket.send_data(message)

        # receive array of outcomes of shape [2, batch_size, N_msmt]
        msmt, done = self.server_socket.recv_data()
        msmt = tf.cast(msmt, tf.float32)
        mask = tf.where(msmt[0]==1, 1.0, 0.0) # [batch_size, N_msmt]

        # calculate reward. If training, include penalty for measuring m1='e'
        Z = - msmt[1] * mask
        if epoch_type == 'training':
            Z -= penalty_coeff * (1-mask)

        z = tf.math.reduce_mean(Z, axis=[1])
        return z

    def reward_stabilizer_remote(self, N_msmt, epoch_type, penalty_coeff,
                                 stabilizer_amplitudes, stabilizer_signs):
        """
        Send the action sequence to remote environment and receive rewards.
        The data received from the remote env should be sigma_z measurement
        outcomes of shape [2, N_stabilizers, N_msmt, batch_size].

        """
        # return 0 on all intermediate steps of the episode
        if self._elapsed_steps != self.episode_length:
            return tf.zeros(self.batch_size, dtype=tf.float32)

        action_batch = {}
        for a in self.history.keys() - ['msmt']:
            # reshape to [batch_size, T, action_dim]
            action_history = np.array(self.history[a][1:])
            action_batch[a] = np.transpose(action_history,
                            axes=[1,0]+list(range(action_history.ndim)[2:]))

        # send action sequence and metadata to remote client
        message = dict(action_batch=action_batch,
                       batch_size=self.batch_size,
                       N_msmt=N_msmt,
                       epoch_type=epoch_type,
                       epoch=self._epoch,
                       stabilizers=stabilizer_amplitudes,
                       stabilizer_signs=stabilizer_signs)

        self.server_socket.send_data(message)

        # receive sigma_z of shape [2 or 3, N_stabilizers, N_msmt, batch_size]
        # first dimension is interpreted as m1, m2, & m0 (optionally)
        msmt, done = self.server_socket.recv_data()
        msmt = tf.cast(msmt, tf.float32)
        mask = tf.where(msmt[0]==1, 1.0, 0.0) # [N_stabilizers, N_msmt, batch_size]

        m2 = msmt[1] * np.reshape(stabilizer_signs, [len(stabilizer_signs),1,1])
        # mask out trajectories where m1 != 'g'
        Z = np.ma.array(m2, mask = np.where(msmt[0]==1, 0.0, 1.0))

        # mask out trajectories where m0 != 'g'
        if msmt.shape[0] == 3:
            Z = np.ma.array(Z, mask = np.where(msmt[2]==1, 0.0, 1.0))

        z = np.mean(Z, axis=(0,1))

        # If training, include penalty for measuring m1='e'
        if epoch_type == 'training':
            z -= penalty_coeff * (1-np.mean(mask, axis=(0,1)))

        return tf.cast(z, tf.float32)


    @tf.function
    def reward_zero(self, *args):
        """
        Reward is always zero (use when not training).

        """
        return tf.zeros(self.batch_size, dtype=tf.float32)




    ### PROPERTIES

    @property
    def init(self):
        return self._init

    @init.setter
    def init(self, val):
        try:
            assert val in ['vac']
            self._init = val
        except:
            raise ValueError('Initial state not supported.')

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, size):
        # if 'code_map' in self.__dir__():
        #     raise ValueError('Cannot change batch_size after initialization.')
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
