"""
IQ mixer calibration script by Phil Reinhold. 

Before running this:
    1) Setup LO generators in get_LO_frequency()
    2) Specify the fpga modes
"""
import numpy as np
import time
from fpga_lib.get_instruments import get_instruments
from fpga_lib.scripting import get_experiment, silence_logging
instruments = get_instruments()
from instrumentserver.instrument_plugins.SA124B import saCloseDevice, \
    saGetSerialNumberList, saOpenDeviceBySerialNumber, saReadSingleFreqIQ
import atexit
from scipy.optimize import fmin, brute

print('Creating spec')
err, spec_id = saOpenDeviceBySerialNumber(saGetSerialNumberList()[0])
atexit.register(lambda: saCloseDevice(spec_id))
print('Created spec')

# (Card number, channel number, amplitude)
modes = [
    (0, 0, 0.33), # qubit g-e
    (0, 2, 0.70), # readout
    (1, 0, 0.50)  # storage
]


# Maximum # of iterations in optimization
MAXITER = 50
# Terminate optimization if this many iterations were within FLAT_DELTA_POWER
TERM_IF_N_FLAT = 20
# Power level considered flat
FLAT_DELTA_POWER = 3
ACCEPT_LEAKAGE = -95
ACCEPT_SIDEBAND = -95
# If the power level is above BRUTE_POW, use a brute force search to start
BRUTE_POW = -90 #-51
# Number of points in brute force search is N_BRUTE^2
N_BRUTE = 6
# Range for brute force search
BRUTE_MAX_OFS = 10000#2000
# Reference level for the signal hound
GUESS_LEVEL = -60
# BW for signal hound
SPEC_BW = 20e3
# N averages for signal hound
SPEC_N_AVG = 100000
# Whether or not to do a sweep to identify the peak frequency
DO_PEAK_SCAN = False
PEAK_SCAN_RANGE = 6e2
PEAK_SCAN_PTS = 21


# the offset variable on yng only accteps values of 0 or 1 for the channel
def map_chan(chan):
    if chan == 0 or chan==1:
        return 0
    if chan == 2 or chan==3:
        return 1


def get_LO_frequency(card, chan):
    """
    Tells the script how to get carrier frequencies for different (card, chan)
    """
    if card == 0 and chan==0:
        carr = instruments['qubit_LO'].get_frequency()
    elif card == 0 and chan==2:
        carr = instruments['readout_LO'].get_frequency()
    elif card == 1 and chan==0:
        carr = instruments['storage_LO'].get_frequency()
    else:
        raise ValueError((card, chan))
    return carr


def main():
    final_data = []
    for card, chan, amp in modes:
        print('='*25)
        print('Card %d, Channel %d, amp %.3f' % (card, chan, amp))
        print('='*25)
        play_constant(card, chan, amp=amp)
        carrier = get_LO_frequency(card, chan)
        print('Test leakage')
        _, final_leakage_pow = opt_leakage(card, map_chan(chan), carrier)
        print('Test sideband')
        _, final_sideband_pow = opt_sideband(card, chan, carrier)

        final_data.append((final_sideband_pow, final_leakage_pow))
    print('='*50)
    print('Summary: (Card Chan Amp (leakage_pw, sideband_pw))')
    print('='*50)
    for (card, chan, amp), info in zip(modes, final_data):
        print(card, chan, amp, info)
    print('='*50)


def get_power_at(freq):
    return saReadSingleFreqIQ(spec_id, freq, bandwidth=SPEC_BW, 
                              guess_level=GUESS_LEVEL, n_avg=SPEC_N_AVG)


def play_constant(card, chan, amp=0.1):
    exp = get_experiment('fpga_std.test.constant')
    silence_logging(exp)
    exp.card = card
    exp.chan = chan
    exp.amp = amp
    exp.digital = False
    exp.delay = 0
    exp.length = 1000
    exp.run()


def min_power_at(freq, init_params, param_setter, accept_pow, brute_pow, 
                 param_range, test_step):
    print('Minimize at freq:', freq)

    class OptFoundException(Exception):
        def __init__(self, params):
            super(OptFoundException, self).__init__()
            self.params = params

    check_flat = False
    _last_pows = []
    def cost(params):
        param_setter(params)
        pow = get_power_at(freq)
        print(params, pow)
        if pow < accept_pow:
            print('Acceptable power reached')
            raise OptFoundException(params)
        if check_flat:
            _last_pows.append(pow)
            if len(_last_pows) > TERM_IF_N_FLAT:
                if np.all(abs(np.array(_last_pows) - pow) < FLAT_DELTA_POWER):
                    print('Power no longer sufficently decreasing')
                    raise OptFoundException(params)
                _last_pows.pop(0)
        return pow

    try:
        if cost(init_params) >= brute_pow:
            init_params = brute(cost, param_range, Ns=N_BRUTE, finish=None)

        check_flat = True
        new_params = fmin(cost, init_params, maxiter=MAXITER)

    except OptFoundException as e:
        new_params = e.params

    pow0 = get_power_at(freq)
    param_setter(new_params + test_step)
    pow1 = get_power_at(freq)
    param_setter(new_params)
    assert pow1 - pow0 > 18, 'Insufficient power reduction: %.1f, %.1f' % (pow0, pow1)

    return new_params, pow0


def opt_leakage(card, chan, carrier, accept_pow=ACCEPT_LEAKAGE, brute_pow=BRUTE_POW):
    """ Optimize LO leakage. """
    freq = carrier
    peak_freq = find_peak(freq)
    init_offset = get_offset(card, chan)
    param_ranges = [(-BRUTE_MAX_OFS, BRUTE_MAX_OFS)] * 2
    test_step = np.array([5000, 5000])
    return min_power_at(
        peak_freq, init_offset, offset_setter(card, chan), 
        accept_pow, brute_pow, param_ranges, test_step
    )


def opt_sideband(card, chan, carrier, accept_pow=ACCEPT_SIDEBAND, brute_pow=BRUTE_POW):
    """ Optimize sideband leakage. """
    freq = carrier - get_sideband_freq(card, chan)
    peak_freq = find_peak(freq)
    init_params = get_sideband_params(card, chan)
    param_ranges = [(.75, 1.5), (-180, 180)]
    test_step = np.array([0, 180])
    return min_power_at(
        peak_freq, init_params, sideband_setter(card, chan), 
        accept_pow, brute_pow, param_ranges, test_step
    )


def get_sideband_freq(card, chan):
    return instruments['yng%d' % card].get('ssbfreq%d' % chan)


def get_offset(card, chan):
    return instruments['yng%d' % card].get('offset%d' % chan)


def get_sideband_params(card, chan):
     yng = instruments['yng%d' % card]
     return [yng.get('ssbratio%d' % chan), yng.get('ssbtheta%d' % chan)]


def sideband_setter(card, chan):
    yng = instruments['yng%d' % card]
    def set_sideband(params):
        getattr(yng, 'set_ssbratio%d' % chan)(params[0])
        getattr(yng, 'set_ssbtheta%d' % chan)(params[1])
        yng.update_modes()
        time.sleep(.3)
    return set_sideband


def offset_setter(card, chan):
    yng = instruments['yng%d' % card]
    def set_offset(params):
        getattr(yng, 'set_offset%d' % chan)(params)
        yng.update_modes()
        time.sleep(.3)
    return set_offset


def find_peak(f0, range=PEAK_SCAN_RANGE, pts=PEAK_SCAN_PTS):
    """
    Sweep frequency near f0 to identify the peak in power.
    """
    if not DO_PEAK_SCAN:
        return f0
    print('Finding peak near %.3g GHz' % (f0 / 1e9))
    freqs = np.linspace(f0-range, f0+range, pts)
    pows = []
    for f in freqs:
        p = get_power_at(f)
        print('%.2f, %d' % (p, (f-f0)))
        pows.append(p)
    fopt = freqs[np.argmax(pows)]
    print('Optimum at detune = %.3g Hz' % (fopt-f0))
    return fopt


if __name__ == '__main__':
    main()
