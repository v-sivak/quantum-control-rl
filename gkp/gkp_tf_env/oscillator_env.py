# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:30:01 2020

@author: Vladimir Sivak
"""

from gkp.gkp_tf_env.gkp_tf_env import GKP

class OscillatorGKP(GKP):
    
    def __init__(self, **kwargs):
        self.tensorstate = False
        super(OscillatorGKP, self).__init__(**kwargs)
