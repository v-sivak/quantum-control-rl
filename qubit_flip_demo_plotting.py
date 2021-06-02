# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:55:37 2021

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_config
import os
import sys
from matplotlib.animation import FuncAnimation


folder_name = r'E:\data\gkp_sims\PPO\simple_demo\3'
log = np.load(os.path.join(folder_name, 'data.npz'))

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create animation with gaussian policies changing as epochs go on, also 
# showing which policies were sampled from these gaussians.
fig, ax = plt.subplots(1, 1, figsize=(3.375,2), dpi=200)
savename = os.path.join(folder_name, r'gaussian.gif')
plt.tight_layout()
ax.set_ylim(0, 14)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'Action, $a$')
ax.set_ylabel('Probability density')
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.plot(np.ones(2)*0.5, [0,14], linestyle='--', color='k')
ax.plot(-np.ones(2)*0.5, [0,14], linestyle='--', color='k')
xs = np.linspace(-1, 1, 301)

epochs = len(log['mean'])
batch = len(log['train_actions'][0])

line, = ax.plot(xs, np.zeros_like(xs))
line2 = ax.plot(np.arange(batch), np.zeros(batch), 
                linestyle='none', marker='.', color='brown')[0]

def gaussian(x, _mean, _sigma):
    return 1/np.sqrt(2*np.pi*_sigma**2) * np.exp(-(x-_mean)**2/2/_sigma**2)

def update(i):
    label = 'Epoch {0}'.format(i+1)
    print(label)
    ys = gaussian(xs, log['mean'][i], log['sigma'][i])
    
    points = log['train_actions'][i]
    vals = gaussian(points, log['mean'][i], log['sigma'][i])
    line2.set_ydata(vals)
    line2.set_xdata(points)
    
    line.set_ydata(ys)
    ax.set_title(label, fontsize=9)
    return line, line2, ax

anim = FuncAnimation(fig, update, frames=np.arange(0, epochs-1), interval=200)
anim.save(savename, writer='imagemagick', dpi=600)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create animation with gaussian policies changing as epochs go on, also 
# showing which policies were sampled from these gaussians.
fig, ax = plt.subplots(1, 1, figsize=(3.375,2), dpi=200)
savename = os.path.join(folder_name, r'gaussians.pdf')
plt.tight_layout()
ax.set_ylim(0, 14)
ax.set_xlim(-1, 1)
ax.set_xlabel(r'Action, $a$')
ax.set_ylabel('Probability density')
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.plot(np.ones(2)*0.5, [0,14], linestyle='--', color='k')
ax.plot(-np.ones(2)*0.5, [0,14], linestyle='--', color='k')

epochs = len(log['mean'])
batch = len(log['train_actions'][0])

def gaussian(x, _mean, _sigma):
    return 1/np.sqrt(2*np.pi*_sigma**2) * np.exp(-(x-_mean)**2/2/_sigma**2)

palette = plt.get_cmap('tab10')

for i, e in enumerate([0, 20, 30, 40, 50]):
    xs = np.linspace(-1, 1, 301)
    ys = gaussian(xs, log['mean'][e], log['sigma'][e])
    ax.plot(xs, ys, color=palette(i))
    e_points = e if e<50 else 49
    points = log['train_actions'][e_points]
    vals = gaussian(points, log['mean'][e], log['sigma'][e])
    ax.plot(points, vals, linestyle='none', marker='.', color=palette(i),
            label=e)
ax.legend()
fig.savefig(savename, dpi=600)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create animaiton with measurement outcomes (0 or 1) appearing in each epoch
fig, ax = plt.subplots(1,1, figsize=(10,5))
savename = os.path.join(folder_name, r'bits.gif')
plt.axis('off')
plt.tight_layout()

def update(i):
    string = str(((1+log['train_rewards'][i])/2).astype(int))
    string = string.replace(' ', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = '\n'.join(string)
    
    string0 = string.replace('1', '')
    string1 = string.replace('0', '')
    
    ax.text(0.02*i, 0.5, string0, va='center', color='red')
    ax.text(0.02*i, 0.5, string1, va='center', color='green')

anim = FuncAnimation(fig, update, frames=np.arange(0, epochs-1), interval=200)
anim.save(savename, writer='imagemagick', dpi=600, 
          savefig_kwargs={'transparent': True, 'facecolor': 'none'})


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create a static figure with measurement outcomes, similar to prev animation
fig, ax = plt.subplots(1,1, figsize=(5,2), dpi=200)
savename = os.path.join(folder_name, r'bits.png')
plt.axis('off')
plt.tight_layout()
for i in range(epochs-1):
    string = str(((1+log['train_rewards'][i])/2).astype(int))
    string = string.replace(' ', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    ax.text(0.05*i,0,'\n'.join(string), va='center')
fig.savefig(savename, bbox_inches='tight')


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create a static figure with measurement outcomes as colored squares
fig, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=200)
savename = os.path.join(folder_name, r'bit_map.pdf')
# plt.axis('off')
plt.tight_layout()
vals = np.where(log['train_rewards']==1, 0.6, -0.9)

ax.pcolormesh(np.transpose(vals), cmap='RdYlGn', vmin=-1, vmax=1)

ax.set_xticks(np.arange(0, vals.shape[0], 1), minor=True)
ax.set_yticks(np.arange(0, vals.shape[1], 1), minor=True)
ax.grid(which='both', color='w', linestyle='-')
ax.set_aspect('equal')

ax.set_ylabel('Batch')
ax.set_xlabel('Epoch')
fig.savefig(savename, dpi=600)

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Create animation showing how Gradient Ascent would work on this problem
fig, ax = plt.subplots(1,1, figsize=(3.375,2))
ax.set_xlabel('Iteration, k')
ax.set_ylabel('Fidelity')
ax.set_xlim(0,20)
ax.set_ylim(0,1)
# ax.set_ylim(1e-5,1)
# ax.set_yscale('log')
ax.set_xticks([0,5,10,15,20])
savename = os.path.join(folder_name, r'GD.gif')
plt.tight_layout()

a = [0.02]
lr = 0.9e-2
steps = 22

for i in range(steps):
    a_new = a[-1] + lr * 2*np.pi*np.sin(2*np.pi*a[-1])
    a.append(a_new)
a = np.array(a)

fidelity = np.sin(np.pi*a)**2
fidelity_true = np.sin(np.pi*a**1.2)**2

# line2 = ax.plot(np.zeros(2), np.zeros(2), color='red')[0]
line1 = ax.plot(np.zeros(2), np.zeros(2), color='green')[0]
text1 = ax.text(12, 0.5, '$a$=%.3f'%a[0])
text2 = ax.text(12, 0.35, 'F=%.3f'%fidelity[0])
# text3 = ax.text(12, 0.2, 'F=%.3f'%fidelity[0], color='red')

def update(i):
    i = min(i, steps-1)
    line1.set_xdata(np.arange(i))
    line1.set_ydata(fidelity[:i])
    # line2.set_xdata(np.arange(i))
    # line2.set_ydata(fidelity_true[:i])
    text1.set_text('a=%.3f'%a[i])
    text2.set_text('F=%.5f'%fidelity[i])
    # text3.set_text('F=%.5f'%fidelity_true[i])
    return line1, text1, text2

anim = FuncAnimation(fig, update, frames=np.arange(0, steps+10), interval=300)
anim.save(savename, writer='imagemagick', dpi=600)



# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cretae a static figure showing sample complexity required for fidelity
# estimation on a static protocol, and sample complexity of RL training 
# required to achieve the same fidelity. It demonstrates that the agent can
# reach close to the shot-noise limit.
folder_name = r'E:\data\gkp_sims\PPO\simple_demo\sweep11'
all_trajectories = []
N_trajectories = len(os.listdir(folder_name))-1
for i in range(N_trajectories):
    log = np.load(os.path.join(folder_name, 'data'+str(i)+'.npz'))
    sigma_z = -log['eval_rewards'].squeeze()
    fidelity = (1-sigma_z)/2
    all_trajectories.append(fidelity)
eval_epochs = log['eval_epochs']
batch = len(log['train_rewards'][0])

fig, ax = plt.subplots(1,1, figsize=(3.375,2.5), dpi=200)
savename = os.path.join(folder_name, 'sample_complexity.png')
ax.set_ylim(1e-5,1.2)
# ax.set_xlim(0,60)
ax.set_yscale('log')
ax.set_xlabel('Epoch')
ax.set_ylabel('Infidelity')
for i in range(N_trajectories):
    ax.plot(eval_epochs, 1-all_trajectories[i], alpha=0.3, linestyle='--')

median = np.median(all_trajectories, axis=0)
logmean = 1-np.exp(np.mean(np.log(1-np.array(all_trajectories)),axis=0))

ax.plot(eval_epochs, 1/(1+eval_epochs*batch), color='k', label='shot noise limit')
ax.plot(eval_epochs, 1-median, color='r', linestyle='--', label='median')
ax.plot(eval_epochs, 1-logmean, color='k', linestyle='--', label='logmean')
ax.legend(loc='lower left')

msmt = lambda x: x*batch
ax2 = ax.secondary_xaxis('top', functions=(msmt, msmt))
ax2.set_xlabel('# of measurements, M')
# ax2.set_xticks([0,300,600,900,1200,1500])

plt.tight_layout()

fig.savefig(savename, dpi=600)