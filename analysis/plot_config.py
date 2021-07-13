# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 12:52:25 2020

@author: Vladimir Sivak
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

### Setup matplotlib
fontsize = 8
fontsize_tick = 7
linewidth = 0.75
spinewidth = 1.0
markersize = linewidth*6
tick_size = 3.0
pad = 2

mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['savefig.transparent'] = True
#mpl.rcParams['figure.subplot.bottom'] = 0.2
#mpl.rcParams['figure.subplot.right'] = 0.85
#mpl.rcParams['figure.subplot.left'] = 0.18
mpl.rcParams['axes.linewidth'] = spinewidth
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.labelpad'] = pad

mpl.rcParams['xtick.major.size'] = tick_size
mpl.rcParams['xtick.major.width'] = spinewidth
mpl.rcParams['xtick.minor.size'] = tick_size / 1.5
mpl.rcParams['xtick.minor.width'] = spinewidth / 1.5

mpl.rcParams['ytick.major.size'] = tick_size
mpl.rcParams['ytick.major.width'] = spinewidth
mpl.rcParams['ytick.minor.size'] = tick_size / 1.5
mpl.rcParams['ytick.minor.width'] = spinewidth / 1.5

mpl.rcParams['xtick.major.pad']= pad
mpl.rcParams['ytick.major.pad']= pad
mpl.rcParams['xtick.minor.pad']= pad / 2.0
mpl.rcParams['ytick.minor.pad']= pad / 2.0

mpl.rcParams['xtick.labelsize'] = fontsize_tick
mpl.rcParams['ytick.labelsize'] = fontsize_tick

mpl.rcParams['legend.fontsize'] = fontsize_tick
mpl.rcParams['legend.frameon'] = True

mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['lines.markersize'] = markersize
mpl.rcParams['lines.markeredgewidth'] = linewidth / 2
            
mpl.rcParams['legend.markerscale'] = 2.0