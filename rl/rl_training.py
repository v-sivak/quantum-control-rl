# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:26:46 2021

@author: qulab
"""


from fpga_lib.scripting import get_experiment, wait_complete, get_last_results
from init_script import *


e = get_experiment('gkp_exp.rl.CD_calibration')
e.training_loop()