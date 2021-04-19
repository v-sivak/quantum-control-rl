from simulator import config
from gkp.tf_env import tf_env
from gkp.tf_env import tf_env_wrappers
from gkp.tf_env import policy
from gkp.tf_env import helper_functions
import importlib

def env_init(control_circuit, **kwargs):
    # Load default parameters of oscillator-qubit system
    params = {k: v for k, v in config.__dict__.items() if '__' not in k}
    kwargs = {**params, **kwargs}  # Add/override default params with kwargs
    
    module_name = 'gkp.environments.' + control_circuit
    return importlib.import_module(module_name).QuantumCircuit(**kwargs)