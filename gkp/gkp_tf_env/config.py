#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:09:16 2020

@author: Vladimir Sivak
"""

# Units: seconds, Hz

# Oscillator
T1_osc = 245e-6
T2_osc = None
K_osc = 1

# Qubit
T1_qb = 50e-6
T2_qb = 60e-6

# Coupling
chi = 28e3

# Imperfections
t_gate = 1.2e-6
t_read = 0.6e-6
t_feedback = 0.6e-6
t_idle = 0.

# Simulator discretization
discrete_step_duration = 100e-9
