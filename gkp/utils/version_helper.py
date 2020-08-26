"""
Compatibility shim for some versioning

Created on Tue Aug 25 14:04:38 2020

@author: Henry Liu
"""
from distutils.version import LooseVersion

import tf_agents
from tf_agents.policies import tf_policy

if LooseVersion(tf_agents.__version__) >= "0.6":
    TFPolicy = tf_policy.TFPolicy
else:
    TFPolicy = tf_policy.Base


__all__ = ["TFPolicy"]
