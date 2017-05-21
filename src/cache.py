#!/usr/bin/env python3
import numpy as np
import scipy
from scipy import sparse
import pandas

# we assume all data is in ../data/um/

cats = {1: 'base', 2: 'valid', 3: 'hidden', 4: 'probe', 5: 'qual'}

# save cached arrays of all the data as seperate categories given in cats
def write_arrs():
    print('reading all data')
    data = pandas.read_csv('../data/um/all.dta', header=None, sep=' ', 
        dtype=np.uint32).values
    inds = pandas.read_csv('../data/um/all.idx', header=None, sep=' ', 
        dtype=np.uint8).values[:, 0]
    for i, name in cats.items():
        print('saving', name)
        np.save('../data/um/' + name, data[inds == i])  # select by cat. index

# read a cached category array and return it
def read_arr(name):
    data = np.load('../data/um/' + name + '.npy')
    print(name + ' imported:', data.shape, data.dtype)
    return data

# save cached UsersxMovies matrices of all the data as seperate categories 
def write_mats():
    for i, name in cats.items():
        if name == 'qual':
            continue
        data = read_arr(name)
        mat = np.zeros((458293, 17770), dtype=np.uint8)  # hard code dimensions
        for point in data:
            mat[point[0] - 1, point[1] - 1] = point[3]
        del data  # free memory
        mat = sparse.csc_matrix(mat)
        print('saving', name)
        sparse.save_npz('../data/' + name, mat, compressed=False)
        del mat  # free memory

# read a cached category matrix and return it
def read_mat(name):
    loaded = np.load('../data/' + name + '.npz')
    cls = getattr(scipy.sparse, 'csc_matrix')
    data = cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
    print(name + ' imported mat[user, movie]:', data.shape, data.dtype)
    return data

