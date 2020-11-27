# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 10:32:23 2020

@author: qulab
"""


import numpy as np
import time
from fpga_lib.get_instruments import get_instruments
from fpga_lib.scripting import get_experiment, silence_logging
instruments = get_instruments()
from instrumentserver.instrument_plugins.SA124B import saCloseDevice, \
    saGetSerialNumberList, saOpenDeviceBySerialNumber, saReadSingleFreqIQ
import atexit
import matplotlib.pyplot as plt

from leakage_opt_fpga import play_constant, get_sideband_freq, \
    get_LO_frequency

print('Creating spec')
err, spec_id = saOpenDeviceBySerialNumber(saGetSerialNumberList()[0])
atexit.register(lambda: saCloseDevice(spec_id))
print('Created spec')

# (Card number, channel number)
card, chan = (1, 0)

# DAC amplitudes
start, stop, points = (0.01, 0.9, 10)

# spec params
SPEC_BW = 20e3
SPEC_N_AVG = 100000
GUESS_LEVEL = -60

# before SA input
attenuation = -30-20-3.5

if __name__ == '__main__':

    carrier_freq = get_LO_frequency(card, chan)
    sideband_freq = carrier_freq + get_sideband_freq(card, chan)
    
    amplitudes = 10 ** np.linspace(np.log10(start), np.log10(stop), points)
    output_powers = []
    for amp in amplitudes:
        play_constant(card, chan, amp=amp)
        power = saReadSingleFreqIQ(spec_id, sideband_freq, bandwidth=SPEC_BW, 
                              guess_level=GUESS_LEVEL, n_avg=SPEC_N_AVG)
        output_powers.append(power)        
        time.sleep(2)

    input_powers = 20*np.log10(amplitudes)
    output_powers = np.array(output_powers) - attenuation
    
    fig, axes = plt.subplots(2,1, sharex=True)
    axes[0].set_ylabel('Output power (dBm)')
    axes[0].plot(input_powers, output_powers)
    axes[1].set_xlabel('DAC power (uncalibrated, max=0)')
    axes[1].set_ylabel('Gain')
    axes[1].plot(input_powers, output_powers-input_powers)
    plt.tight_layout()
    