import numpy as np
from cache import *
import pandas


def read_files(directory, files):
    A = []
    for file in files:
        A.append(pandas.read_csv(directory + file + '.dta', header=None, 
            sep=' ', dtype=np.float32).values[:, 0])
    return np.array(A, dtype=np.float32)


def probe_blend(probe_files, qual_files=None, save_name='p_blend', directory='../data/blend0/'):
    # note that qual files and probe files must be in the same order!
    print('Blending on probe...')
    A = read_files(directory, probe_files).T
    s = read_arr('probe')[:, 3].astype(np.float32)

    print('Getting pseudo-inverse...')
    alphas = np.matmul(np.linalg.pinv(A), s)

    print('\nAlphas:')
    for m, model in enumerate(probe_files):
        print('%s: %f' % (model, alphas[m]))

    print('Probe Error:', np.sqrt(np.mean((np.dot(alphas, A.T) - s) ** 2)))

    if not qual_files:
        return

    A = read_files(directory, qual_files)

    print('\nMaking submission...')
    submission = np.dot(alphas, A)
    submission = np.zeros(submission.shape, dtype=np.float32)
    np.savetxt(directory + save_name + '.dta', submission, fmt='%.3f', newline='\n')
    print('Finished! saved submission.')


def mean_blend(sub_files, save_name='mean_blend', directory='../data/blend0/'):
    submission = np.mean(read_files(directory, sub_files), axis=0)
    np.savetxt(directory + save_name + '.dta', submission, fmt='%.3f', newline='\n')
    print('Finished! saved submission.')


def qual_blend(qual_files, save_name='q_blend', directory='../data/blend0/'):
    # note that qual files and probe files must be in the same order!
    print('Blending on qual...')
    A = read_files(directory, qual_files)
    s = read_arr('qual')[:, 3].astype(np.float32)

    zeros = (3.84358 ** 2.) * 2749898.
    print('Getting alphas...')
    inv = np.linalg.inv(np.matmul(A, A.T))
    ats = 0.5 * (zeros + np.sum(A ** 2, axis=1))
    alphas = np.dot(inv, ats)

    print('\nAlphas:')
    for m, model in enumerate(qual_files):
        print('%s: %f' % (model, alphas[m]))

    print('\nMaking submission...')
    submission = np.dot(alphas, A)
    np.savetxt(directory + save_name + '.dta', submission, fmt='%.3f', newline='\n')
    print('Finished! saved submission.')



if __name__ == '__main__':
    pass
    # probe_blend(['p-nnfactor', 'p-avg'], ['q-nnfactor', 'q-avg'], directory='../data/blend0/')
    # mean_blend(['1', '2', '3', '4'], directory='../data/blend1/')
    # qual_blend(['1', '2', '3', '4'], directory='../data/blend1/')


