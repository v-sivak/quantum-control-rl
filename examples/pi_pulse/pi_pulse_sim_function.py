# Author: Ben Brock 
# Created on May 03, 2023 

import numpy as np
import qutip as qt

def pi_pulse_sim(amp, # amp of cos^2 pulse in GHz
                 drag, # dimensionless drag param
                 detuning, # detuning of pi pulse in GHz
                 t_duration = 40, # duration of pi pulse in ns
                 N = 5, # number of transmon states
                 kerr = -0.1, # kerr nonlinearity of transmon in GHz
                 n_times = 101 # number of time-steps for the qutip simulation
                 ):

    q = qt.destroy(N)
    psi0 = qt.fock(N,0)

    # frequencies in GHz, times in ns
    ts = np.linspace(-t_duration/2,t_duration/2,n_times)

    args = {'amp':amp,
            't_duration':t_duration,
            'drag':drag,
            'kerr':kerr}

    H0 = -2*np.pi*detuning*(q.dag()*q) + 2*np.pi*kerr*(q.dag()**2)*(q**2)
    H1 = q.dag()
    H2 = q

    def H1_coeff(t,args):
        omega_pulse = (np.pi)/args['t_duration']
        pulse = args['amp']*(np.cos(omega_pulse*t)**2)
        pulse = pulse + 1j*(args['drag']/(2*args['kerr']))*args['amp']*omega_pulse*np.sin(2*omega_pulse*t)
        return pulse

    def H2_coeff(t,args):
        omega_pulse = (np.pi)/args['t_duration']
        pulse = args['amp']*(np.cos(omega_pulse*t)**2)
        pulse = pulse - 1j*(args['drag']/(2*args['kerr']))*args['amp']*omega_pulse*np.sin(2*omega_pulse*t)
        return pulse

    H = [H0,[H1,H1_coeff],[H2,H2_coeff]]
    result = qt.sesolve(H,psi0,tlist=ts,args=args)
    this_reward = (2*qt.expect(qt.fock_dm(N,1),result.states[-1]))-1 # return reward as pauli measurement
    return this_reward
        