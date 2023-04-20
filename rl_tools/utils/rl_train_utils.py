import h5py


def save_log(log, filename, groupname):
    """
    Simple log saver.

    Inputs:
        log -- dictionary of arrays or lists
        filename -- name of the .hdf5 file in which to save the log
        groupname -- group name for this log

    """
    h5file = h5py.File(filename, 'a')
    try:
        grp = h5file.create_group(groupname)
        for name, data in log.items():
            grp.create_dataset(name, data=data)
    finally:
        h5file.close()
