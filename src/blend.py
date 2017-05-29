#!/usr/bin/env python3

import numpy as np
from cache import *
import pandas


def read_files(directory, files):
    A = []
    for file in files:
        A.append(pandas.read_csv(directory + file, header=None, 
            sep=' ', dtype=np.float32).values[:, 0])
    return np.array(A, dtype=np.float32)


def probe_blend(probe_files, qual_files=None, save_name='p_blend', directory='blend0'):
    # note that qual files and probe files must be in the same order!
    print('Blending on probe...')
    # directory = '../data/submissions/' + directory + '/'
    A = read_files(directory, probe_files).T
    s = read_arr('probe')[:, 3].astype(np.float32)

    print('Getting pseudo-inverse...')
    alphas = np.matmul(np.linalg.pinv(A), s)

    print('\nAlphas:')
    for m, model in enumerate(probe_files):
        print('{:60}: {: f}'.format(model, alphas[m]))

    print('\nProbe Error:', np.sqrt(np.mean((np.dot(alphas, A.T) - s) ** 2)))

    if not qual_files:
        return

    A = read_files(directory, qual_files)

    print('\nMaking submission...')
    submission = np.dot(alphas, A)
    np.savetxt(directory + save_name + '.dta', submission, fmt='%.3f', newline='\n')
    print('Finished! saved submission.')


def mean_blend(sub_files, save_name='mean_blend', directory='blend0'):
    directory = '../data/submissions/' + directory + '/'
    submission = np.mean(read_files(directory, sub_files), axis=0)
    np.savetxt(directory + save_name + '.dta', submission, fmt='%.3f', newline='\n')
    print('Finished! saved submission.')


def qual_blend(qual_files, save_name='q_blend', directory='blend0'):
    # note that qual files and probe files must be in the same order!
    print('Blending on qual...')
    directory = '../data/submissions/' + directory + '/'
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
    files = [
            # global
            ('average_probe', 'average_qual'),
            # KNNs
            ('probe-KNN-', 'qual-KNN-'),
            ('final_unprocessed_probe_pred_50.dta', 'final_unprocessed_qual_pred_50.dta'),
            # SVDs
            # overfit
            ('OVERFIT_0.949640_50_0.010000_0.030000_150_probe.txt',  'OVERFIT_0.949640_50_0.010000_0.030000_150_qual.txt'),
            ('OVERFIT_0.949723_100_0.010000_0.030000_100_probe.txt', 'OVERFIT_0.949723_100_0.010000_0.030000_100_qual.txt'),
            # biases
            ('BEST_NAIVE_SVD_0.917174_50_0.010000_0.030000_100_probe.txt', 'BEST_NAIVE_SVD_0.917174_50_0.010000_0.030000_100_qual.txt'),
            ('0.919819_75_0.007000_0.050000_100_probe.txt',                               '0.919819_75_0.007000_0.050000_100_pred.txt'),
            # something....
            ('probe22-F=50-NR=98291669-NB=30-SD',           'output22-F=50-NR=98291669-NB=30-SD'),
            # timesvd++
            ('probe24-F=10-NR=98291669-NB=5-SD-TBS-Time',   'output24-F=10-NR=98291669-NB=5-SD-TBS-Time'),
            ('probe10-F=30-NR=98291669-NB=5-SD-TBS-Time',   'output10-F=30-NR=98291669-NB=5-SD-TBS-Time'),
            ('probe18-F=60-NR=98291669-NB=5-SD-TBS-Time',   'output18-F=60-NR=98291669-NB=5-SD-TBS-Time'),
            ('probe14-F=250-NR=98291669-NB=5-SD-TBS-Time',  'output14-F=250-NR=98291669-NB=5-SD-TBS-Time'),
            ('probe17-F=60-NR=98291669-NB=15-SD-TBS-Time',  'output17-F=60-NR=98291669-NB=15-SD-TBS-Time'),
            ('probe24-F=30-NR=98291669-NB=5-SD-TBS-Time',   'output24-F=30-NR=98291669-NB=5-SD-TBS-Time'),
            ('probe12-F=200-NR=98291669-NB=15-SD-TBS-Time', 'output12-F=200-NR=98291669-NB=15-SD-TBS-Time'),
            ('probe12-F=250-NR=98291669-NB=1-SD-TBS-Time',  'output12-F=250-NR=98291669-NB=1-SD-TBS-Time')
             ]

    probes = [p for (p, q) in files]
    quals = [q for (p, q) in files]

    assert(len(probes) == len(quals))

    probe_blend(probes, quals, directory='/home/nkgupta/tmp/BLENDING/')
    # mean_blend(['1', '2', '3', '4'], directory='../data/blend1/')
    # qual_blend(quals, directory='/home/nkgupta/tmp/BLENDING/')


