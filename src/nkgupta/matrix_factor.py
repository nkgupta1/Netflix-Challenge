#!/usr/bin/env python3
# creates and factors a matrix

import numpy as np
from prob2utils import train_model, get_err

M = 943     # number of users
N = 1682    # number of movies
K = 20      # number of latent factors

def create_Y():
    '''
    returns an M x N matrix of users and movie ratings
    Currently using all the data (including probe)
    '''

    M = 458293     # number of users
    N = 17770      # number of movies
    num_ratings = 102416306    # number of ratings (including 0s)
    num_qual    = 2749898

    # Y = np.zeros((M, N))
    Y = np.zeros((num_ratings, 3), dtype=np.int)

    with open('../../data/um/all.dta','r') as f:
        rating_count = 0
        for line in f:
            lst = line.split()

            if len(lst) == 0:
                continue

            rating = int(lst[3])
            
            if rating == 0:
                # ignore qual data points
                continue

            user = int(lst[0]) - 1 # Zero-indexing
            movie = int(lst[1]) - 1 # Zero-indexing


            Y[rating_count][0] = user
            Y[rating_count][1] = movie
            Y[rating_count][2] = rating

            rating_count += 1

    return Y

def cross_validate(Y_train, Y_test, regs, etas):
    '''
    cross validates the model, varying regularization strength and step size.
    '''
    print('training size =', len(Y_train))
    print('testing size  =', len(Y_test))
    print()

    for reg in regs:
        for eta in etas:
            U, V, a, b, _ = train(Y_train, reg, eta, Y_test=Y_test, zero_mean=False, save=False)
            errIn = get_err(U, V, a, b, Y_train, reg=0)
            errOut = get_err(U, V, a, b, Y_test, reg=0)
            output_str = ''
            output_str = '{}, errOut = {:.6f}'.format(output_str, errOut)
            output_str = '{}, reg = {:.5f}'.format(output_str, reg)
            output_str = '{}, eta = {:.4f}'.format(output_str, eta)
            output_str = '{}, errIn = {:.6f}'.format(output_str, errIn)
            print(output_str[2:])

def train(Y, reg, eta, Y_test=None, zero_mean=True, save=True):
    '''
    learns U, V, a, b
    '''

    (U, V, a, b, err) = train_model(M, N, K, eta, reg, Y, Y_test=Y_test, eps=0.003)

    if zero_mean:
        V = V - V.mean(axis=0)

    A, S, B = np.linalg.svd(V, full_matrices=False)

    if save:
        np.save('models/{:6.5f}-U-{:.5f}-{:.4f}'.format(err, reg, eta), U)
        np.save('models/{:6.5f}-V-{:.5f}-{:.4f}'.format(err, reg, eta), V)
        np.save('models/{:6.5f}-a-bias-{:.5f}-{:.4f}'.format(err, reg, eta), a)
        np.save('models/{:6.5f}-b-bias-{:.5f}-{:.4f}'.format(err, reg, eta), b)
        np.save('models/{:6.5f}-A-{:.5f}-{:.4f}'.format(err, reg, eta), A[:, :2])

    return U, V, a, b, err

if __name__ == '__main__':

    eta = .01   # step size
    reg = .1    # regularization strength

    remove_mean = True

    etas = [0.1, 0.01, 0.005]
    regs = [1, 0.1, 0.01]

    Y = create_Y()

    if remove_mean:
        Y_mean = Y.mean(axis=0)
        # don't modify the id numbers
        Y_mean[0] = 0
        Y_mean[1] = 0

        Y = Y - Y_mean

    num_samples = len(Y)
    # Y_train = Y[:2*num_samples//3]
    # Y_test  = Y[2*num_samples//3:]

    # Y_train = Y[num_samples//3:]
    # Y_test  = Y[:num_samples//3]

    train(Y, reg, eta)
    # cross_validate(Y_train, Y_test, regs, etas)

    