"""
Spectroscopy experiment to find e-f transition.

ssb_scan only works in the range +-131 MHz because of the finite register size.
Therefore, we need to do a combination of ssb_scan and software detuning to
reach the desired drive frequency given some fixed LO. In the 4-channel mode
there is only +-250 MHz available with our DAC. So one possible option is to
set the LO 50 MHz below the g-e transition (and ssb to +50 MHz for g-e drive).
Then we can scan_ssb in the range [-130, 130 MHz] and if the software detuning
is also set to -50 MHz, then it means that we can go to -130 -50 -50 = -230 MHz
from g-e and to +130 -50 -50 = +30 MHz on the other side. Should be enough for
the qubit with unharmonicity around 200 MHz.

"""

from init_script import *

class qubit_ef_spec(FPGAExperiment):
    detune_range = RangeParameter((-10e6,10e6,41))
    selective = BoolParameter(True)
    fit_func = 'gaussian'
    loop_delay = FloatParameter(1e6)

    def sequence(self):
        mode = chi_drive
        with mode.scan_detune(*self.detune_range):
            delay(2000)
            qubit.flip() # prep e
            sync()
            mode.flip(selective=self.selective)
            readout()
            delay(self.loop_delay)
