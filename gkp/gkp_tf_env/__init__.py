from gkp.gkp_tf_env import config
from gkp.gkp_tf_env import gkp_tf_env
from gkp.gkp_tf_env import tf_env_wrappers
from gkp.gkp_tf_env import policy
from gkp.gkp_tf_env import helper_functions

import gkp.quantum_circuit as qc

def gkp_init(simulate, **kwargs):
    # Load default parameters of oscillator-qubit system
    params = {k: v for k, v in config.__dict__.items() if '__' not in k}
    kwargs = {**params, **kwargs}  # Add/override default params with kwargs

    return qc.__getattribute__(simulate).QuantumCircuit(**kwargs)