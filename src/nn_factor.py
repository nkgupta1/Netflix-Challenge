import tensorflow as tf
import numpy as np
from cache import *
import keras
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense, Dropout


class NN:
    def __init__(self, mode='both', K=100, epochs=3):
        self.num_users = 458293
        self.data_mean = 3.60860891887339
        self.K, self.epochs = K, epochs
        # number of blocks to split the data into b/c of memory limitations
        self.num_blocks = 50
        self.read_data()
        if mode == 'train':
            self.train()
            self.save_model()
        elif mode == 'predict':
            self.make_model()
            self.model.load_weights('../models/nn-k20-e20-rmse0.877.h5')  # add model name here
            self.predict('nn20-20')
        elif mode == 'both':
            self.train()
            self.save_model()
            self.predict('nnfac-100e3')


    def read_data(self):
        self.num_samples = self.num_users - (self.num_users % self.num_blocks)
        self.original_base = read_mat('base').astype(np.float32)
        print('splitting....')
        base = self.original_base[:self.num_samples]
        self.remainder = self.original_base[self.num_samples:]
        self.block_size = (self.num_users // self.num_blocks)
        self.base = [base[i * self.block_size:(i + 1) * self.block_size] 
                    for i in range(0, self.num_blocks)]


    def my_mse(self, y_true, y_pred):
        # custom loss function for model
        nonzero = tf.to_float(tf.not_equal(y_true, tf.constant(0, 
            dtype=tf.float32)), name='ToFloat')
        return tf.reduce_mean(tf.multiply(tf.square(tf.subtract(y_true, 
            y_pred)), nonzero))


    def generate_floats(self):
        # generator to get training data for model
        self.num_blocks
        block = 0
        while True:
            data = self.base[block].toarray()
            data[data == 0] = self.data_mean
            data -= self.data_mean
            yield (data, data)
            block = (block + 1) % self.num_blocks


    def make_model(self):
        self.rmse = real_RMSE()
        self.model = Sequential()
        self.model.add(Dense(self.K, input_shape=(17770,), activation='linear'))  # hidden layer
        self.model.add(Dense(17770, activation='linear'))  # output layer
        self.model.add(Dropout(0.4))
        self.model.summary()  # double-check model format
        # 'sgd' optimizer might not be a bad idea instead of adam:
        self.model.compile(loss=self.my_mse, optimizer=optimizers.adam(lr=0.0001))       


    def train(self):
        self.make_model()
        self.model.fit_generator(self.generate_floats(), samples_per_epoch=self.num_samples,
            nb_epoch=self.epochs, verbose=True, callbacks=[self.rmse])
        # note that rmse will be sqrt(my_mse / sparsity) = srqt(my_mse * 80)


    def save_model(self):
        self.model.save_weights('../models/nn-k' + str(self.K) + '-e' + str(self.epochs) 
            + '-rmse' + str(self.rmse.losses[-1])[:5] + '.h5')


    def predict(self, save_name):
        print('predicting from model...')
        qual = read_arr('qual')
        qual_ratings = []
        block = 0
        new_block_start = 1
        for p, point in enumerate(qual):
            if point[0] >= new_block_start:
                print('predicting next block. finished %.3f percent of predictions' 
                    % (float(p) / len(qual)))
                old_block_start = new_block_start
                new_block_start += self.block_size
                if block >= self.num_blocks:
                    data = self.remainder.toarray()
                else:
                    data = self.base[block].toarray()
                block += 1
                data[data == 0] = self.data_mean
                data -= self.data_mean
                prediction = self.model.predict(data)
            qual_ratings.append(prediction[point[0] - old_block_start, point[1] - 1])
        qual_ratings = np.array(qual_ratings, dtype=np.float32) + self.data_mean

        # save predictions
        np.savetxt('../data/' + save_name + '.dta', qual_ratings, fmt='%.3f', newline='\n')
        print('finished!')



# metric callback class to print custom RMSE for model
class real_RMSE(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.sparsity = 86.4229420728  # sparsity of data in full matrix
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        rmse = np.sqrt(self.sparsity * logs.get('loss'))
        print('\nReal RMSE:', rmse)
        self.losses.append(rmse)



if __name__ == '__main__':
    NN()



'''
Notes: 
Submitting all of the mean is   RMSE: 1.12882 (-18.65% above water)
Submitting k=5 epochs=1 is      RMSE: 1.12577 (-18.33% above water)
Submitting k=20 epochs=5 is     RMSE: 1.05539 (-10.93% above water) [predicted 0.94 training]
Submitting k=50 epochs=50 is    RMSE: 0.98403 (-3.43% above water) [predicted 0.80 training]
Submitting k=100 epochs=50 is   RMSE: 0.99372 (-4.45% above water) [predicted 0.75 training]
Submitting k=100 epochs=20 is   RMSE: 0.9936 (-4.44% above water) [predicted 0.80 training]
Submitting k=50 epochs=20 is    RMSE: 0.99717 (-4.81% above water) [predicted 0.84 training]
Submitting k=200 epochs=50 is   RMSE: 1.01436 (-6.62% above water) [predicted 0.67 training]
Submitting k=20 epochs=20 is    RMSE: 1.00924 (-6.08% above water) [predicted 0.88 training]
'''