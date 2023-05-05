# Author: Ben Brock 
# Created on May 03, 2023 
#%%
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


N = 5 # number of transmon levels
n_times = 101
t_duration = 40
q = qt.destroy(N)
psi0 = qt.fock(N,0)

# frequencies in GHz, times in ns
kerr = -0.1
ts = np.linspace(-t_duration/2,t_duration/2,n_times)

omega_pulse = (np.pi)/t_duration
pulse_array_real = ((np.pi)/t_duration)*(np.cos(omega_pulse*ts)**2)
pulse_array_imag = np.zeros_like(pulse_array_real)

H0 = 2*np.pi*kerr*(q.dag()**2)*(q**2)
H1 = q.dag()
H2 = q

pulse = pulse_array_real + 1j*pulse_array_imag
pulse_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse)
pulse_conj_func = qt.interpolate.Cubic_Spline(ts[0], ts[-1], pulse.conj())

fig,ax = plt.subplots(1,1)
ax.plot(ts,pulse_func(ts).real*1e3,label='real')
ax.plot(ts,pulse_func(ts).imag*1e3,label='imag')
ax.set_xlabel('t (ns)')
ax.set_ylabel('pulse amp (MHz)')
ax.legend()
plt.show()


H = [H0,[H1,pulse_func],[H2,pulse_conj_func]]

result = qt.sesolve(H,psi0,tlist=ts)


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
