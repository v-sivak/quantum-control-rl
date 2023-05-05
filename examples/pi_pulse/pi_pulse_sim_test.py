# Author: Ben Brock 
# Created on May 03, 2023 
#%%
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from pi_pulse_sim_function import pi_pulse_sim

N = 5 # number of transmon levels
n_times = 101
t_duration = 40
q = qt.destroy(N)
psi0 = qt.fock(N,0)

# frequencies in GHz, times in ns
kerr = -0.1
ts = np.linspace(-t_duration/2,t_duration/2,n_times)

amp = (np.pi)/t_duration
drag = 0.5
detuning = 0.0

args = {'amp':amp,
        't_duration':t_duration,
        'drag':drag,
        'kerr':kerr}

H0 = -2*np.pi*detuning*(q.dag()*q) + 2*np.pi*kerr*(q.dag()**2)*(q**2)
H1 = q
H2 = q.dag()

e_ops = [q.dag()*q]

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


fig,ax = plt.subplots(1,1)
ax.plot(ts,np.real(H1_coeff(ts,args))*1e3,label='real')
ax.plot(ts,np.imag(H1_coeff(ts,args))*1e3,label='imag')
ax.set_xlabel('t (ns)')
ax.set_ylabel('pulse amp (MHz)')
ax.legend()
plt.show()


H = [H0,[H1,H1_coeff],[H2,H2_coeff]]

result = qt.sesolve(H,psi0,tlist=ts,args=args)



probs = np.zeros((N,n_times),dtype=float)
fig,ax = plt.subplots(1,1)
for ii in range(N):
    for jj in range(n_times):
        probs[ii,jj] = qt.expect(qt.fock_dm(N,ii),result.states[jj])

    ax.plot(ts,probs[ii],label=str(ii))
ax.set_xlabel('t (ns)')
ax.set_ylabel('P(n)')
ax.legend()
plt.show()

fig,ax = plt.subplots(1,1)
ax.plot(ts,probs[2])
ax.set_xlabel('t (ns)')
ax.set_ylabel('P(2)')
ax.set_yscale('log')
plt.show()

fig,ax = plt.subplots(1,1)
ax.plot(ts,1-probs[1])
ax.set_xlabel('t (ns)')
ax.set_ylabel('Infidelity')
ax.set_yscale('log')
plt.show()

# %%
