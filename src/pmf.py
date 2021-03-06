#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
from cache import *

class PMF:
    def __init__(self, num_feat=120, epsilon=0.005, _lambda=0.1, momentum=0.8, 
            maxepoch=30, num_batches=10, batch_size=100000):
        self.num_feat = num_feat
        self.epsilon = epsilon
        self._lambda = _lambda
        self.momentum = momentum
        self.maxepoch = maxepoch
        self.num_batches = num_batches
        self.batch_size = batch_size
        
        self.w_C = None
        self.w_I = None

        self.err_train = []
        self.err_val = []
        
    def fit(self, train_vec, val_vec): 
        # mean subtraction
        self.mean_inv = np.mean(train_vec[:,2])
        
        pairs_tr = train_vec.shape[0]
        pairs_va = val_vec.shape[0]
        
        # 1-p-i, 2-m-c
        num_inv = int(max(np.amax(train_vec[:,0]), np.amax(val_vec[:,0]))) + 1
        num_com = int(max(np.amax(train_vec[:,1]), np.amax(val_vec[:,1]))) + 1

        incremental = False
        if ((not incremental) or (self.w_C is None)):
            # initialize
            self.epoch = 0
            self.w_C = 0.1 * np.random.randn(num_com, self.num_feat)
            self.w_I = 0.1 * np.random.randn(num_inv, self.num_feat)
            
            self.w_C_inc = np.zeros((num_com, self.num_feat))
            self.w_I_inc = np.zeros((num_inv, self.num_feat))
        
        
        while self.epoch < self.maxepoch:
            self.epoch += 1

            # Shuffle training truples
            shuffled_order = np.arange(train_vec.shape[0])
            np.random.shuffle(shuffled_order)

            # Batch update
            for batch in range(self.num_batches):
                # print "epoch %d batch %d/%d" % (self.epoch, batch+1, self.num_batches)

                batch_idx = np.mod(np.arange(self.batch_size * batch,
                                             self.batch_size * (batch+1)),
                                   shuffled_order.shape[0])

                batch_invID = np.array(train_vec[shuffled_order[batch_idx], 0], dtype='int32')
                batch_comID = np.array(train_vec[shuffled_order[batch_idx], 1], dtype='int32')

                # Compute Objective Function
                pred_out = np.sum(np.multiply(self.w_I[batch_invID,:], 
                                                self.w_C[batch_comID,:]),
                                axis=1) # mean_inv subtracted

                rawErr = pred_out - train_vec[shuffled_order[batch_idx], 2] + self.mean_inv

                # Compute gradients
                Ix_C = 2 * np.multiply(rawErr[:, np.newaxis], self.w_I[batch_invID,:]) \
                        + self._lambda * self.w_C[batch_comID,:]
                Ix_I = 2 * np.multiply(rawErr[:, np.newaxis], self.w_C[batch_comID,:]) \
                        + self._lambda * self.w_I[batch_invID,:]
            
                dw_C = np.zeros((num_com, self.num_feat))
                dw_I = np.zeros((num_inv, self.num_feat))

                # loop to aggreate the gradients of the same element
                for i in range(self.batch_size):
                    dw_C[batch_comID[i],:] += Ix_C[i,:]
                    dw_I[batch_invID[i],:] += Ix_I[i,:]


                # Update with momentum
                self.w_C_inc = self.momentum * self.w_C_inc + self.epsilon * dw_C
                self.w_I_inc = self.momentum * self.w_I_inc + self.epsilon * dw_I


                self.w_C = self.w_C - self.w_C_inc
                self.w_I = self.w_I - self.w_I_inc

                # Compute Objective Function after
                if batch == self.num_batches - 1:
                    inds = (np.random.rand(10000) * train_vec.shape[0]).astype(np.uint32)
                    pred_out = np.sum(np.multiply(self.w_I[np.array(train_vec[inds,0], dtype='int32'),:],
                                                    self.w_C[np.array(train_vec[inds,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - train_vec[inds, 2] + self.mean_inv
                    obj = LA.norm(rawErr) ** 2 \
                            + 0.5*self._lambda*(LA.norm(self.w_I) ** 2 + LA.norm(self.w_C) ** 2)

                    self.err_train.append(np.sqrt(np.mean(rawErr ** 2)))

                # Compute validation error
                if batch == self.num_batches - 1:
                    inds = (np.random.rand(10000) * val_vec.shape[0]).astype(np.uint32)
                    pred_out = np.sum(np.multiply(self.w_I[np.array(val_vec[inds,0], dtype='int32'),:],
                                                    self.w_C[np.array(val_vec[inds,1], dtype='int32'),:]),
                                        axis=1) # mean_inv subtracted
                    rawErr = pred_out - val_vec[inds, 2] + self.mean_inv
                    self.err_val.append(np.sqrt(np.mean(rawErr ** 2)))

                # Print info
                if batch == self.num_batches - 1:
                    print 'Training RMSE: %f, Test RMSE %f' % (self.err_train[-1], self.err_val[-1])
                    self.save_name = '-e%d-t%.3f-v%.3f' % (self.epoch, self.err_train[-1], self.err_val[-1])
                    self.submission('qual')
                    self.submission('probe')


    def predict(self, invID): 
        return np.dot(self.w_C, self.w_I[invID,:]) + self.mean_inv
        
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            self.num_feat = parameters.get("num_feat", 10)
            self.epsilon = parameters.get("epsilon", 1)
            self._lambda = parameters.get("_lambda", 0.1)
            self.momentum = parameters.get("momentum", 0.8)
            self.maxepoch = parameters.get("maxepoch", 20)
            self.num_batches = parameters.get("num_batches", 10)
            self.batch_size = parameters.get("batch_size", 1000)

    def submission(self, source):
        print('predicting from model...')
        qual = read_arr(source)
        # qual[:, :2] -= 1
        qual_ratings = []
        user = -1
        self.num_users = 458293
        for p, point in enumerate(qual):
            if point[0] != user:
                user = point[0]
                movie_vec = self.predict(user) + 3.60860891887339
            qual_ratings.append(movie_vec[point[1]])
        qual_ratings = np.array(qual_ratings, dtype=np.float32)
        print(qual_ratings.shape)

        # save predictions
        np.savetxt('../data/submissions/pmf-' + source + str(self.save_name) + '.dta', 
            qual_ratings, fmt='%.3f', newline='\n')
        print('finished!')


    def read_data(self):
        self.train = self.preprocess_data(read_arr('base'))
        self.test = self.preprocess_data(read_arr('probe'))
        self.num_batches = self.train.shape[0] / self.batch_size
        print('num batches:', self.num_batches)


    def preprocess_data(self, arr):
        arr = np.delete(arr, 2, 1).astype(np.float32) 
        arr[:, 2] -= 3.60860891887339
        return arr


if __name__ == '__main__':
    pmf = PMF()
    pmf.read_data()
    pmf.fit(pmf.train, pmf.test)
    pmf.submission()

    # pmf = PMF()
    # train = np.array([[1, 0, 5],[1,2,3],[1,3,1],[1,4,5]]).astype(np.float32)
    # test = train.copy()
    # test[:, 2] += 0.05
    # pmf.fit(train, test)

    # print(pmf.predict(1))





