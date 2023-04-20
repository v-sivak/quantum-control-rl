from rl_tools.tf_env import tf_env
from rl_tools.tf_env import tf_env_wrappers
from rl_tools.tf_env import policy
from rl_tools.tf_env import helper_functions
import importlib

def env_init(control_circuit, **kwargs):
    kwargs = {**kwargs}  # Add/override default params with kwargs

    module_name = 'rl_tools.environments.' + control_circuit
    return importlib.import_module(module_name).QuantumCircuit(**kwargs)
