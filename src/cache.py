import numpy as np
from scipy import sparse

# read a cached npy matrix from path
def np_read(path='../data/um/', name='all'):
    data = np.load(path + name + '.npy')
    print(name + ' imported:', data.shape, data.dtype)
    return data

# read a .dta file from path and cache npy matrix in its place
def np_write(path='../data/um/', name='all'):
    with open(path + name + '.dta') as f:
        data = [[int(x) for x in line.split()] for line in f.readlines()]
        data = np.array(data, dtype=np.uint32)
        np.save(path + name, data)

# read a cached npy array of data from path and write a scipy sparse 
# UsersxMovies matrix 
def sp_mat_write(path='../data/', data=None):
    if data is None:
        data = np_read()
    data = data[data[:, 3] != 0]
    num_users = np.max(data[:, 0])
    num_movies = np.max(data[:, 1])
    mat = np.zeros((num_users, num_movies), dtype=np.uint8)
    for point in data:
        mat[point[0] - 1, point[1] - 1] = point[3]
    del data  # free memory
    csc = sparse.csc_matrix(mat)
    del mat  # free memory
    sparse.save_npz(path + 'csc', csc, compressed=False)

# read a cached scipy sparse UsersxMovies matrix of all the data
def sp_mat_read(path='../data/'):
    data = sparse.load_npz(path + 'csc.npz')
    print(' imported csc matrix[user, movie]:', data.shape, data.dtype)
    return data
