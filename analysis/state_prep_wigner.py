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


exp_dir = r'D:\DATA\exp\gkp_exp.state_prep.state_prep_wigner_tomography\archive'
fname = '20210413.h5'
group = 19
file_name = os.path.join(exp_dir, fname)

grp_name = str(group)
res = Results.create_from_file(file_name, grp_name)
wigner_full = res['m2_postselected_m0_m1'].data
wigner = np.mean(wigner_full, axis=0)

wigner_fname = r'Y:\tmp\for Vlad\from_vlad\wigner_fock1_epoch300.npz'
np.savez(wigner_fname, wigner=wigner.data)