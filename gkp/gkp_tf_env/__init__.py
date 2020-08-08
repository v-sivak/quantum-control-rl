from gkp.gkp_tf_env import config
from gkp.gkp_tf_env import gkp_tf_env
from gkp.gkp_tf_env import tf_env_wrappers
from gkp.gkp_tf_env import policy
from gkp.gkp_tf_env import helper_functions
from gkp.gkp_tf_env import oscillator_env
from gkp.gkp_tf_env import oscillator_qubit_env


def gkp_init(simulate, **kwargs):
    # Load default parameters of oscillator-qubit system
    params = {k: v for k, v in config.__dict__.items() if '__' not in k}
    kwargs = {**params, **kwargs}  # Add missing defaults to kwargs

    if simulate == 'oscillator':
        return oscillator_env.OscillatorGKP(**kwargs)
    elif simulate == 'oscillator_qubit':
        return oscillator_qubit_env.OscillatorQubitGKP(**kwargs)
