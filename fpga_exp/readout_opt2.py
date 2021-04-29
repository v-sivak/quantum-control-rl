from init_script import *

class readout_opt2(ReadoutOptV2Experiment):
    readout_instance = readout
    loop_delay = IntParameter(1e6)
    n_regions = IntParameter(3)
    cool_qubit = BoolParameter(True)

    def sequence(self):

        for i in range(self.n_regions):
            if self.cool_qubit:
                system.cool_qubit()
                sync()
                delay(1e3)
                sync()
            self.prepare(i)
            sync()
            delay(24)
            sync()
            self.readout_instance(traj='rel', se='se')
            sync()
            delay(self.loop_delay)

    def prepare(self,i):
        if i ==0: #g
            sync()
        if i==1: #e
            sync()
            qubit.flip()
        if i==2: #f
            pass
#            sync()
#            qubit_alice.flip()
#            sync()
#            qubit_alice.ef_pulse()
