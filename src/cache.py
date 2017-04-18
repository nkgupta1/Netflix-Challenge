import numpy as np

def np_read(path='../data/um/', name='all'):
    data = np.load(path + name + '.npy')
    print(name + ' imported:', data.shape, data.dtype)
    return data

def np_write(path='../data/um/', name='all'):
    with open(path + name + '.dta') as f:
        data = [[int(x) for x in line.split()] for line in f.readlines()]
        data = np.array(data, dtype=np.uint32)
        np.save(path + name, data)