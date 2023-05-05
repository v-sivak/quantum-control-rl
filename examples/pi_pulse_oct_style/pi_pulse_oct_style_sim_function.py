# Author: Ben Brock 
# Created on May 03, 2023 

import numpy as np
import qutip as qt

def pi_pulse_oct_style_sim(pulse_array_real, # real pulse
                            pulse_array_imag, # imag pulse
                            N = 5, # number of transmon states
                            kerr = -0.1, # kerr nonlinearity of transmon in GHz
                            n_times = 101 # number of time-steps for the qutip simulation
                            ):

    q = qt.destroy(N)
    psi0 = qt.fock(N,0)

    
    # frequencies in GHz, times in ns
    t_duration = len(pulse_array_real)
    ts = np.linspace(-t_duration/2,t_duration/2,n_times)

    H0 = 2*np.pi*kerr*(q.dag()**2)*(q**2)
    H1 = q.dag()
    H2 = q

    pulse = pulse_array_real + 1j*pulse_array_imag
    pulse_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse)
    pulse_conj_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse.conj())

    H = [H0,[H1,pulse_func],[H2,pulse_conj_func]]
    result = qt.sesolve(H,psi0,tlist=ts)
    this_reward = (2*qt.expect(qt.fock_dm(N,1),result.states[-1]))-1 # return reward as pauli measurement
    return this_reward
        