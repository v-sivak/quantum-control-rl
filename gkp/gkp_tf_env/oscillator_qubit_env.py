# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:36 2020

@author: Vladimir Sivak
"""

from gkp.gkp_tf_env.gkp_tf_env import GKP

class OscillatorQubitGKP(GKP):
    
    def __init__(self, **kwargs):
        self.tensorstate = True
        super(OscillatorQubitGKP, self).__init__(**kwargs)
