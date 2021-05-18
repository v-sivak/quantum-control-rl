"""
Sweep ScriptedPolicy across some timing param and fit T1s

Created on Fri Aug 28 16:32:56 2020

@author: Henry Liu
"""
import os
import io
from time import time
from datetime import datetime
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, pi, exp
from scipy.optimize import curve_fit

from rl_tools.tf_env import helper_functions as hf
from rl_tools.tf_env import env_init
from rl_tools.tf_env import policy as plc


def add_plot_to_zip(fig, filename, savepath, zipname):
    with ZipFile(os.path.join(savepath, "%s.zip" % zipname), "a") as z:
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        z.writestr(os.path.join(zipname, "%s.png" % filename), buf.getvalue())
        buf.close()


class ActionScript(object):

    def __init__(self):

        delta = 0.19
        eps = 0.19
        self.period = 6
        b_amp = sqrt(8*pi/sqrt(3))
        a_amp = sqrt(2*pi/sqrt(3))

        self.script = {
            'alpha' : [1j*a_amp*exp(-0j*pi/3), delta*exp(-2j*pi/3),
                       1j*a_amp*exp(-2j*pi/3), delta*exp(-1j*pi/3),
                       1j*a_amp*exp(-1j*pi/3), delta*exp(-0j*pi/3)],
            'beta'  : [1j*b_amp*exp(-2j*pi/3), -eps*exp(-2j*pi/3),
                       1j*b_amp*exp(-1j*pi/3), -eps*exp(-1j*pi/3),
                       1j*b_amp*exp(-0j*pi/3), -eps*exp(-0j*pi/3)],
            'phi' : [pi/2]*6,
            'theta' : [0]*6
            }


#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

savepath = "."
filename = datetime.now().isoformat(timespec="seconds").replace(":", "_")
params = np.linspace(0, 5e-6, 11, dtype=np.float32)
states = ['Z+']

with ZipFile(os.path.join(savepath, "%s.zip" % filename), "a") as z:
    z.writestr(os.path.join(filename, ""), b"")

#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

lifetimes = np.zeros(len(params))
returns = np.zeros(len(params))

gfig, gax = plt.subplots(1,1, dpi=300, figsize=(10,6))
gax.set_title(r'Reward curves')
gax.set_ylabel(r'Reward')
gax.set_xlabel('Time')

for j in range(len(params)):
    t = time()
    env = env_init(control_circuit='oscillator_qubit',
               init='Z+', H=1, batch_size=2500, episode_length=200,
               reward_mode = 'fidelity', quantum_circuit_type='v2',
               encoding = 'hexagonal', t_feedback = params[j])
    action_script = ActionScript()
    policy = plc.ScriptedPolicy(env.time_step_spec(), action_script)

    for state in states:
        if '_env' in env.__dir__():
            env._env.init = state
        else:
            env.init = state

        # Collect batch of episodes
        time_step = env.reset()
        policy_state = policy.get_initial_state(env.batch_size)
        rewards = np.zeros((env.episode_length, env.batch_size))
        counter = 0
        while not time_step.is_last()[0]:
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state
            time_step = env.step(action_step.action)
            rewards[counter] = time_step.reward
            counter += 1

        # Fit T1
        mean_rewards = rewards.mean(axis=1) # average across episodes
        returns[j] = np.sum(mean_rewards)
        times = np.arange(env.episode_length)*env.step_duration
        T1_guess = (times[-1]-times[0])/(mean_rewards[0]-mean_rewards[-1])
        popt, pcov = curve_fit(hf.exp_decay, times, mean_rewards,
                               p0=[1, T1_guess])
        lifetimes[j] = popt[1]*1e6

        # Add plots
        fig, ax = plt.subplots(1,1, dpi=300, figsize=(10,6))
        ax.set_title(r'Reward for param=$%s$, fit=%f, %f' % (params[j], *popt))
        ax.set_ylabel(r'Reward')
        ax.set_xlabel('Time')
        ax.plot(times, hf.exp_decay(times, *popt), "r-")
        ax.scatter(times, mean_rewards)

        # Add fit to global
        gax.plot(times, hf.exp_decay(times, *popt), label=j, color=str(1/(len(params) * 2) * j + 0.5))

        with ZipFile(os.path.join(savepath, "%s.zip" % filename), "a") as z:
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            z.writestr(os.path.join(filename, "reward-%i.png" % j), buf.getvalue())
            buf.close()

            buf = io.BytesIO()
            np.save(buf, mean_rewards)
            z.writestr(os.path.join(filename, "reward-%i.npy" % j), buf.getvalue())
            buf.close()

            buf = io.BytesIO()
            np.save(buf, times)
            z.writestr(os.path.join(filename, "time-%i.npy" % j), buf.getvalue())
            buf.close()

            buf = io.BytesIO()
            np.save(buf, popt)
            z.writestr(os.path.join(filename, "fit-%i.npy" % j), buf.getvalue())
            buf.close()

        plt.close(fig)

    print('(%d): Time %.3f sec' %(j, time()-t), flush=True)

# Plot summary of the sweep and save the sweep data
fig, ax = plt.subplots(1,1, dpi=300, figsize=(10,6))
ax.set_title(r'Logical lifetime')
ax.set_ylabel(r'$T_Z$')
ax.set_xlabel('Sweep parameter')
ax.plot(params, lifetimes)
add_plot_to_zip(fig, "summary", savepath, filename)
plt.close(fig)

fig, ax = plt.subplots(1,1, dpi=300, figsize=(10,6))
ax.set_title(r'Mean reward')
ax.set_ylabel(r'Mean reward')
ax.set_xlabel('Sweep parameter')
ax.plot(params, returns)
add_plot_to_zip(fig, "returns", savepath, filename)
plt.close(fig)

add_plot_to_zip(gfig, "reward-curves", savepath, filename)
plt.close(gfig)

with ZipFile(os.path.join(savepath, "%s.zip" % filename), "a") as z:
    data = {"params": params, "lifetimes": lifetimes, "returns": returns}
    for k, v in data.items():
        buf = io.BytesIO()
        np.save(buf, v)
        z.writestr(os.path.join(filename, "%s.npy" % k), buf.getvalue())
        buf.close()
