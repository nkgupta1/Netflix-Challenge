#!/usr/bin/env python3
# creates and factors a matrix

import numpy as np
import time
from prob2utils import train_model, get_err

M = 458293     # number of users
N = 17770      # number of movies
K = 30         # number of latent factors

def predictions(err, reg, eta, mean=3.51259997602):
    '''
    generate list of predictions
    '''

    print('loading data')

    U = np.load('models/{:6.5f}-U-{:.5f}-{:.4f}.npy'.format(err, reg, eta))
    V = np.load('models/{:6.5f}-V-{:.5f}-{:.4f}.npy'.format(err, reg, eta))
    a = np.load('models/{:6.5f}-a-bias-{:.5f}-{:.4f}.npy'.format(err, reg, eta))
    b = np.load('models/{:6.5f}-b-bias-{:.5f}-{:.4f}.npy'.format(err, reg, eta))

    print('done loading data')

    with open('../../data/um/qual.dta') as f1:
        with open('../../data/submissions/svd-5-epochs-{:6.5f}-{:.5f}-{:.4f}.txt'.format(err, reg, eta), 'w') as f2:
            for line in f1:
                lst = line.split()

                if len(lst) == 0:
                    continue

                user = int(lst[0]) - 1 # Zero-indexing
                movie = int(lst[1]) - 1 # Zero-indexing

                pred = np.dot(U[user-1], V[:,movie-1]) + a[user-1] + b[movie-1] + mean

                if pred < 1:
                    pred = 1
                if pred > 5:
                    pred = 5

                f2.write(str(pred) + '\n')


def create_Y():
    '''
    returns an M x N matrix of users and movie ratings
    Currently using all the data (including probe)
    '''

    num_ratings = 102416306    # number of ratings (including 0s)
    num_qual    = 2749898

    # Y = np.zeros((M, N))
    Y = np.zeros((num_ratings, 3), dtype=np.int)

    print('reading in data')
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

            if rating_count % 5000000 == 0:
                print('done reading', rating_count, 'points')

            user = int(lst[0]) - 1 # Zero-indexing
            movie = int(lst[1]) - 1 # Zero-indexing
            # date = int(lst[2])


            Y[rating_count][0] = user
            Y[rating_count][1] = movie
            Y[rating_count][2] = rating

            rating_count += 1

    print('done reading data')

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
            start = time.time()

            U, V, a, b, _ = train(Y_train, reg, eta, Y_test=Y_test, save=False)
            errIn = get_err(U, V, a, b, Y_train, reg=0)
            errOut = get_err(U, V, a, b, Y_test, reg=0)
            output_str = ''
            output_str = '{}, errOut = {:.6f}'.format(output_str, errOut)
            output_str = '{}, reg = {:.5f}'.format(output_str, reg)
            output_str = '{}, eta = {:.4f}'.format(output_str, eta)
            output_str = '{}, errIn = {:.6f}'.format(output_str, errIn)
            print(output_str[2:])
            print('{:5.0f}\n'.format(time.time() - start))

def train(Y, reg, eta, Y_test=None, save=True):
    '''
    learns U, V, a, b
    '''

    (U, V, a, b, err) = train_model(M, N, K, eta, reg, Y, Y_test=Y_test, eps=0.005, max_epochs=20)

    if save:
        np.save('models/{:6.5f}-U-{:.5f}-{:.4f}'.format(err, reg, eta), U)
        np.save('models/{:6.5f}-V-{:.5f}-{:.4f}'.format(err, reg, eta), V)
        np.save('models/{:6.5f}-a-bias-{:.5f}-{:.4f}'.format(err, reg, eta), a)
        np.save('models/{:6.5f}-b-bias-{:.5f}-{:.4f}'.format(err, reg, eta), b)
        # np.save('models/{:6.5f}-A-{:.5f}-{:.4f}'.format(err, reg, eta), A[:, :2])

    return U, V, a, b, err

if __name__ == '__main__':

    
    # reg = .1    # regularization strength
    # eta = .01   # step size

    # remove_mean = True

    # regs = [1, 0.1, 0.01]
    # etas = [0.1, 0.01, 0.001]

    # Y = create_Y()

    # if remove_mean:
    #     Y_mean = Y.mean(axis=0)
    #     # don't modify the id numbers
    #     Y_mean[0] = 0
    #     Y_mean[1] = 0

    #     print("Remove mean out of Y: ", Y_mean[2])

    #     Y = Y - Y_mean

    # num_samples = len(Y)
    # # Y_train = Y[:2*num_samples//3]
    # # Y_test  = Y[2*num_samples//3:]

    # # Y_train = Y[num_samples//3:]
    # # Y_test  = Y[:num_samples//3]

    # train(Y, reg, eta)
    # cross_validate(Y_train, Y_test, regs, etas)
    

    predictions(0.38454, 0.10000, 0.01000)

    