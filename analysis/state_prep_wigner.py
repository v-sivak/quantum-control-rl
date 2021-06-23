# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 13:30:27 2021

@author: qulab
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from fpga_lib import *
from fpga_lib.dsl.result import Results

exp_dir = r'D:\DATA\exp\gkp_exp.gkp_qec.sbs_feedback_reset_wigner\archive'
#exp_dir = r'D:\DATA\exp\gkp_exp.state_prep.state_prep_wigner_tomography\archive'
fname = '20210623.h5'
group = 0
file_name = os.path.join(exp_dir, fname)

grp_name = str(group)
res = Results.create_from_file(file_name, grp_name)
wigner_full_g = res['g_sz_postselected_m0_m1_m2'].data
wigner_full_e = res['e_sz_postselected_m0_m1_m2'].data
wigner_g = np.mean(wigner_full_g, axis=0)
wigner_e = np.mean(wigner_full_e, axis=0)
xs, ys = res['g_m0'].ax_data[1:]

wigner_fname = r'Y:\tmp\for Vlad\from_vlad\wigner_sbs.npz'
np.savez(wigner_fname, wigner_g=wigner_g.data, wigner_e=wigner_e.data, xs=xs, ys=ys)


wigner_exp = (wigner_g + wigner_e) / 2.0

# Plot 2D Wigner
fig, ax = plt.subplots(1, 1, sharey=True, dpi=300)
fig.suptitle('Wigner (not normalized)')
ax.set_aspect('equal')

plot_kwargs = dict(cmap='RdBu_r', vmin=-np.max(wigner_exp), vmax=np.max(wigner_exp))
p = ax.pcolormesh(xs, ys, np.transpose(wigner_exp), **plot_kwargs)
plt.colorbar(p)
plt.tight_layout()
