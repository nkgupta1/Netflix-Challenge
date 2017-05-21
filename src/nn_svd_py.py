import tensorflow as tf
import numpy as np
from cache import *
from old_cache import *
import keras
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense


class NN:
    def __init__(self, a, b, u, v, mode='svd', epochs=100, layer1=128, layer2=None):
        # users are rows of u. movies are columns of v.
        # a is bias of all users. b is bias of all movies.
        self.data_mean = 3.60860891887339
        self.epochs, self.layer1, self.layer2 = epochs, layer1, layer2
        # number of blocks to split the data into b/c of memory limitations
        self.num_blocks = 100
        # mode = 'predict'
        if mode == 'train':
            self.read_data(a, b, u, v, read_training=True)
            self.train()
            self.save_model()
        elif mode == 'predict':
            self.read_data(a, b, u, v, read_training=False)
            self.make_model()
            self.model.load_weights('../models/nnsvd-k20-e1-rmse1.220.h5')  # add model name here
            self.predict(save_name='nnsvd0')
        elif mode == 'both':
            self.read_data(a, b, u, v, read_training=True)
            self.train()
            self.save_model()
            self.predict()
        elif mode == 'svd':
            self.read_data(a, b, u, v, read_training=True)
            self.get_svd_rmse()


    def read_data(self, a, b, u, v, read_training):
        # takes in file names of sub matrices
        print('reading data...')
        self.a, self.b, self.u, self.v = [np.load('../data/svd/' + x) for x in [a, b, u, v]]
        self.v = self.v.T
        self.K = self.u.shape[1]
        if not read_training:
            return
        all_data = np_read()
        self.num_points = all_data.shape[0]
        print('preprocessing data...')
        original_ij, original_ratings = self.preprocess_data(all_data)
        del all_data
        print('splitting data...')
        # remove a and b biases appropriately from ratings
        self.num_samples = self.num_points - (self.num_points % self.num_blocks)
        self.block_size = (self.num_points // self.num_blocks)
        self.blocks_ij = self.split_data(original_ij)
        self.blocks_ratings = self.split_data(original_ratings)
        self.num_blocks += 1  # because of remainder added


    def preprocess_data(self, data):
        ij = (data[:, :2] - 1) # make the data zero indexed
        if data.shape[1] == 4:
            ratings = (data[:, 3].astype(np.float32)- self.data_mean)
            ratings -= self.a[ij[:, 0]]
            ratings -= self.b[ij[:, 1]]
            ratings = np.expand_dims(ratings, axis=1)
        else:
            ratings = None
        return ij, ratings


    def split_data(self, arr):
        blocks = arr[:self.num_samples]
        remainder = arr[self.num_samples:]
        blocks = [blocks[i * self.block_size:(i + 1) * self.block_size] 
                    for i in range(0, self.num_blocks)]
        blocks.append(remainder)
        return blocks


    def generator_block(self):
        # generator to get training data for model
        block = 0
        while True:
            ij = self.blocks_ij[block]
            trainx = np.concatenate((self.u[ij[:, 0]], self.v[ij[:, 1]]), axis=1)
            ratings = self.blocks_ratings[block]
            print('rmse theoretical:', 
                np.mean((ratings[:, 0] - np.sum(self.u[ij[:, 0]] * self.v[ij[:, 1]], axis=1)) ** 2) ** 0.5)            # trainx = concatenation of user, movie vectors
            yield trainx, ratings
            block = (block + 1) % self.num_blocks


    def make_model(self):
        self.model = Sequential()
        # hidden layer 1
        self.model.add(Dense(self.layer1, input_shape=(self.K * 2,), activation='linear'))
        # hidden layer 2
        if self.layer2:
            self.model.add(Dense(self.layer2, activation='relu'))    
        self.model.add(Dense(1, activation='relu'))  # output layer
        self.model.summary()  # double-check model format
        # 'sgd' optimizer might not be a bad idea instead of adam:
        self.model.compile(loss='mse', optimizer=optimizers.adam()) #lr=0.001   


    def train(self):
        self.make_model()
        self.rmse = RMSE()
        self.model.fit_generator(self.generator_block(), samples_per_epoch=self.num_points,
            nb_epoch=self.epochs, verbose=True, callbacks=[self.rmse])
        # note that rmse will be sqrt(my_mse / sparsity) = srqt(my_mse * 80)


    def save_model(self):
        self.model.save_weights('../models/nnsvd-k' + str(self.K) + '-e' + str(self.epochs) 
            + '-layers' + str(self.layer1) + ',' + str(self.layer2) + '-rmse' 
            + str(self.rmse.losses[-1])[:5] + '.h5')


    def predict(self, save_name='latest-nn+svd'):
        print('predicting from model...')
        print('preprocessing data...')
        qual_ij, _ = self.preprocess_data(read_arr('qual'))
        print('getting vectors...', qual_ij.shape)
        testx = np.concatenate((self.u[qual_ij[:, 0]], self.v[qual_ij[:, 1]]), axis=1)
        print('making predictions...', testx.shape)
        qual_ratings = self.model.predict(testx)[:, 0]
        # undo processing with self.a, self.b
        print('adjusting predictions...', qual_ratings.shape)
        qual_ratings += (self.data_mean + self.a[qual_ij[:, 0]] + self.b[qual_ij[:, 1]])

        # save predictions
        print('saving predictions...', qual_ratings.shape)
        np.savetxt('../data/' + save_name + '.dta', qual_ratings, fmt='%.3f', newline='\n')
        print('finished!')


    def get_svd_rmse(self):
        print('getting svd training rmse...')
        # get training rmse
        rmse_avg = []
        for block in range(0, self.num_blocks):
            ij = self.blocks_ij[block]
            prediction = np.sum(self.u[ij[:, 0]] * self.v[ij[:, 1]], axis=1)
            ratings = self.blocks_ratings[block][:, 0]
            print(self.u.shape, self.v.shape, ij[:10, 0])
            print(ij[:10], self.u[ij[:10, 0]], self.v[ij[:10, 1]]) 
            print(ratings[:10]) # -2.55887055
            print(prediction[:10])
            quit()
            print(prediction, ratings)
            rmse = np.sqrt(np.mean((prediction - ratings) ** 2))
            print('rmse', rmse)
            rmse_avg.append(rmse)
        print('avg rmse', np.mean(rmse_avg))
        # generate submisison for validation rmse
        # self.model = predictor
        # self.predict()



# metric callback class to print custom RMSE for model
class RMSE(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        rmse = np.sqrt(logs.get('loss'))
        print('\nRMSE:', rmse)
        self.losses.append(rmse)



if __name__ == '__main__':
    NN(a='0.36929-a-bias-0.10000-0.0100.npy', 
        b='0.36929-b-bias-0.10000-0.0100.npy',
        u='0.36929-U-0.10000-0.0100.npy',
        v='0.36929-V-0.10000-0.0100.npy')



'''
Notes: 
Submitting all of the mean is   RMSE: 1.12882 (-18.65% above water)
'''