#!/usr/bin/env python3

"""
Code shamelessly borrowed from internet - see below.
"""

"""
RBM.py
Contains the class definition of a Restricted Boltzmann Machine designed to make
predictions on items based on user ratings
Author: Forest Thomas
Notes:
    This RBM is designed to be used for sparse, two-dimensional inputs
    Therefore, most input vectors are going to be two dimensional
    eg: input[m][r]
        -m is the movie (or item)
        -r is the rating for that item
    The sparsity is handled by ignoring inputs with no ratings, therefore the calculations
    will be dynamic based on which movies have been rated.
"""

import numpy as np
import pickle
from math import e
from random import random

class RBM:
    def __init__(self, V, F, K, rate):
        """
        Initializes the RBM
        V - length of the input vector
        F - the number of hidden units
        K - the number of possible ratings
        rate - the learning rate
        """
        #set number of visible and hidden nodes
        self.F = F
        self.K = K
        self.V = V
        
        #initialize Weight Matrix with normally distributed values
        # self.W = []
        # for i in range(V):
        #     self.W.append([])
        #     for j in range(F):
        #         self.W[i].append([])
        #         for k in range(K):
        #             self.W[i][j].append(normalvariate(0, .01))
        self.W = np.random.normal(scale=.01,size=(V,F,K))
                    
        #initialize biases
        # self.vis_bias = []
        # for i in range(V):
        #     self.vis_bias.append([])
        #     for k in range(K):
        #         self.vis_bias[i].append(0)
        self.vis_bias = np.zeros((V,K))

        # self.hid_bias = []
        # for j in range(F):
        #     self.hid_bias.append(0)
        self.hid_bias = np.zeros(F)

        #set learning rate
        self.eps = rate

    def learn_batch(self, T, V):
        """
        Does T steps of T-step contrastive divergence
        T - The number of steps in CD
        V - the set of input vectors
        return - None
        """
        # initialization
        # del_W = []
        # for i in range(self.V):
        #     del_W.append([])
        #     for j in range(self.F):
        #         del_W[i].append([])
        #         for k in range(self.K):
        #             del_W[i][j].append(0)
        del_W = np.zeros((self.V,self.F,self.K))

        # del_vis_bias = []
        # for i in range(self.V):
        #     del_vis_bias.append([])
        #     for k in range(self.K):
        #         del_vis_bias[i].append(0)
        del_vis_bias = np.zeros((self.V,self.K))

        # del_hid_bias = []
        # for j in range(self.F):
        #     del_hid_bias.append(0)
        del_hid_bias = np.zeros(self.F)

        # run all vectors in the batch
        for v in V:
            rated = np.nonzero(v)[0]
            v_last = self.rebuild_input(rated, self.get_hidden_states(v, rated))
            h = []
            for t in range(1, T):
                h = self.get_hidden_probabilities(v_last, rated)
                v_last = self.rebuild_input(rated, h)

            #get change in visible bias    
            # for i in rated:
            #     for k in range(self.K):
            #         del_vis_bias[i][k] += self.eps*(v[i][k] - v_last[i][k])
            del_vis_bias[rated] = np.add(del_vis_bias[rated], 
                self.eps*np.subtract(np.array(v)[rated],np.array(v_last)[rated]))

            #get change in hidden bias
            hdata = self.get_hidden_probabilities(v, rated)
            hmodel = self.get_hidden_probabilities(v_last, rated)
            # for j in range(self.F):
            #     del_hid_bias[j] += eps*(hdata[j] - hmodel[j])
            del_hid_bias = np.add(del_hid_bias,self.eps*np.subtract(hdata,hmodel))

            #get changes in weights
            # Looks ugly, but apparently not very slow
            for i in rated:
                for j in range(self.F):
                    for k in range(self.K):
                        del_W[i][j][k] += self.eps*(hdata[j]*v[i][k] - hmodel[j]*v_last[i][k])


        #update weights
        # for i in range(self.V):
        #     for k in range(self.K):
        #         self.vis_bias[i][k] += del_vis_bias[i][k]
        self.vis_bias = np.add(self.vis_bias,del_vis_bias)
        # for j in range(self.F):
        #     self.hid_bias[j] += del_hid_bias[j]
        self.hid_bias = np.add(self.hid_bias,del_hid_bias)
        # for i in range(self.V):
        #     for j in range(self.F):
        #         for k in range(self.K):
        #             self.W[i][j][k] += del_W[i][j][k]
        self.W = np.add(self.W,del_W)


    def get_hidden_probabilities(self, v, rated):
        """
        Gives the probabilities of the hidden layer given an input vector
        v - an input vector
        rated - the movies rated in the input vector
        return - a list of probabilities for the hidden layer
        """
        probs = []
        for j in range(self.F):
            # s = 0
            # for i in rated:
            #     for k in range(self.K):
            #         s += v[i][k]*self.W[i][j][k]

            # s = 0
            # for i in rated:
            #     s += np.dot(v[i],self.W[i][j])

            s = np.sum(np.multiply(v,self.W[:,j,:]))

            probs.append(sig(self.hid_bias[j] + s))

        # probs2 = np.tensordot(v,self.W,axes=([0,0],[1,2]))

        # probs = [0 for j in range(self.F)]
        # for j in range(self.F):
        #     s = 0
        #     for i in rated:
        #         for k in range(self.K):
        #             s += v[i][k]*self.W[i][j][k]
        #     probs[j] = sig(self.hid_bias[j] + s)

        return probs
    
        
    def rebuild_input(self, rated, h):
        """
        Rebuilds the input vector, given the set of hidden states
        returns probabilities of v
        h - The a binary vector representing the hidden layer
        """
        # v = [[] for i in range(self.V)]
        # for i in rated:
        #     for k in range(self.K):
        #         prob = self.vis_bias[i][k]
        #         for j in range(self.F):
        #             prob += h[j]*self.W[i][j][k]
        #         prob = sig(prob)
        #         if prob > random():
        #             v[i].append(1)
        #         else:
        #             v[i].append(0)

        v = np.zeros((self.V,self.K))
        for i in rated:
            for k in range(self.K):
                prob = self.vis_bias[i][k]
                for j in range(self.F):
                    prob += h[j]*self.W[i][j][k]

                # prob = self.vis_bias[i][k]
                # prob += np.dot(h,self.W[i,:,k])
                v[i][k] = sig(prob)

        rand = np.random.uniform(size=(self.V,self.K))
        low_idxs = v <= rand
        high_idxs = v > rand
        v[low_idxs] = 0
        v[high_idxs] = 1

        return v

        
    def get_hidden_states(self, v, rated):
        """
        gives the hidden states of the rbm given an input vector
        v - input vector
        rated - movies rated in the input vector (list of indices)
        return - a binary vector representing the hidden layer
        """
        # h = []
        # for j in range(self.F):
        #     s = 0
        #     for i in rated:
        #         # for k in range(self.K):
        #         #     s += v[i][k]*self.W[i][j][k]
        #         s += np.sum(np.multiply(v[i],self.W[i][j]))

        #     prob = sig(self.hid_bias[j] + s)
        #     if prob > random():
        #         h.append(1)
        #     else:
        #         h.append(0)

        h = np.zeros(self.F)
        for j in range(self.F):
            s = 0
            for i in rated:
                s += np.sum(np.multiply(v[i],self.W[i][j]))

            h[j] = sig(self.hid_bias[j] + s)

        rand = np.random.uniform(size=self.F)
        low_idxs = h <= rand
        high_idxs = h > rand
        h[low_idxs] = 0
        h[high_idxs] = 1

        return h


    def save_RBM(self, filename):
        """
        Saves the current RBM to a file
        filename - the name of the file to save to
        """
        with open(filename, "wb") as fout:
            pickle.dump(self, fout)

    def load_RBM(self, filename):
        """
        Loads an RBM from a file
        filename - the name of the file to load from
        """
        with open(filename, "rb") as fin:
            rbm = pickle.load(fin)
        return rbm
        

    def get_prediction(self, movie, h):
        """
        Gives the prediction of a movie given a hidden layer
        This is calculated by taking Expected value over all ratings for the movie
        movie - the movie to predict
        h - the hidden layer
        return - a prediction for the movie
        """
        prob = 0
        for k in range(self.K):
            s = self.vis_bias[movie][k]
            for j in range(self.F):
                s += h[j]*self.W[movie][j][k]

            # s = self.vis_bias[movie][k]
            # s += np.dot(h,self.W[movie,:,k])

            prob += (sig(s)*(k+1))

        return prob
        
def sig(x):
    """
    Sigmoid function
    x - real number input
    return - sigmoid(x)
    """
    try:
        s = 1 / (1 + e ** -x)
    except OverflowError:
        s = 1
    return s

def getRated(v):
    """
    Gives the indices of movies that hae a rating (i.e. not None) in the vector
    v - vector containing mostly None with numbers
    return - an array of indices that have been rated in the vector
    """
    # rated = []
    # for i in range(len(v)):
    #     if v[i] != None:
    #     # if not np.array_equal(v[i], np.zeros(5)):
    #         rated.append(i)

    # print(rated)
    # x = np.nonzero(v)[0]
    # print(np.nonzero(v)[0])
    # return rated
    return np.nonzero(v)[0]
    # return v[~np.all(v == 0, axis=1)]
    # return [i for i in v.shape if not np.allclose(v[i,:],0)]