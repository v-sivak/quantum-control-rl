from gkp.gkp_tf_env import config
from gkp.gkp_tf_env import gkp_tf_env
from gkp.gkp_tf_env import tf_env_wrappers
from gkp.gkp_tf_env import policy
from gkp.gkp_tf_env import helper_functions

from gkp.quantum_circuit import phase_estimation_osc_v2
from gkp.quantum_circuit import phase_estimation_osc_v1
from gkp.quantum_circuit import phase_estimation_osc_qb_v2
from gkp.quantum_circuit import phase_estimation_osc_qb_v1
from gkp.quantum_circuit import Baptiste_feedback_osc
from gkp.quantum_circuit import Baptiste_feedback_osc_qb
from gkp.quantum_circuit import Baptiste_autonomous_osc
from gkp.quantum_circuit import Baptiste_autonomous_osc_qb

def gkp_init(simulate, **kwargs):
    # Load default parameters of oscillator-qubit system
    params = {k: v for k, v in config.__dict__.items() if '__' not in k}
    kwargs = {**params, **kwargs}  # Add/override default params with kwargs

    if simulate == 'phase_estimation_osc_v2':
        return phase_estimation_osc_v2.QuantumCircuit(**kwargs)
    
    if simulate == 'phase_estimation_osc_v1':
        return phase_estimation_osc_v1.QuantumCircuit(**kwargs)

    if simulate == 'phase_estimation_osc_qb_v2':
        return phase_estimation_osc_qb_v2.QuantumCircuit(**kwargs)

    if simulate == 'phase_estimation_osc_qb_v1':
        return phase_estimation_osc_qb_v1.QuantumCircuit(**kwargs)

    if simulate == 'Baptiste_feedback_osc':
        return Baptiste_feedback_osc.QuantumCircuit(**kwargs)
    
    if simulate == 'Baptiste_feedback_osc_qb':
        return Baptiste_feedback_osc_qb.QuantumCircuit(**kwargs)

    if simulate == 'Baptiste_autonomous_osc':
        return Baptiste_autonomous_osc.QuantumCircuit(**kwargs)

    if simulate == 'Baptiste_autonomous_osc_qb':
        return Baptiste_autonomous_osc_qb.QuantumCircuit(**kwargs)