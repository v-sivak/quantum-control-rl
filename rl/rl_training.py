# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:26:46 2021

@author: qulab
"""

#from state_prep_wigner_reward import state_prep_wigner_reward
#e = state_prep_wigner_reward()
#e.training_loop()

#from state_prep_fock_reward import state_prep_fock_reward
#e = state_prep_fock_reward()
#e.training_loop()

#from sbs_stabilizer_reward import sbs_stabilizer_reward
#e = sbs_stabilizer_reward()
#e.training_loop()

#from gkp_prep_ECD_stabilizers import gkp_prep_ECD_stabilizers
#e = gkp_prep_ECD_stabilizers()
#e.training_loop()

from sbs_Pauli_reward import sbs_Pauli_reward
e = sbs_Pauli_reward()
e.training_loop()