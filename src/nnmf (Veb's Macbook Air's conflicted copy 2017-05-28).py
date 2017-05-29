import tensorflow as tf
import numpy as np
from cache import *
import keras
from keras.models import Sequential, load_model
from keras import optimizers
from keras.layers.core import Dense


class NN:
    def __init__(self, mode='predict', K=50, epochs=100, save_epochs='all'):
        self.num_users = 458293
        self.data_mean = 3.60860891887339
        self.num_movies = 17770
        self.K, self.epochs = K, epochs
        self.save_epochs = save_epochs
        # size of block to split the data into b/c of memory limitations
        self.block_size = 500

        self.read_data()
        self.save_name = 'nnmf-' + str(self.K) + '-' + str(self.epochs)
        # mode = 'predict'
        if mode == 'train':
            self.train()
            self.save_model()
        elif mode == 'predict':
            self.make_model()
            self.model.load_weights('../models/nnmf-k50-e1-rmse1.047.h5')
            self.predict()
        elif mode == 'both':
            self.train()
            self.save_model()
            self.predict()
        elif mode == 'bag':
            pass # not implemented yet


    def read_data(self):
        self.base = read_mat('base_sub')


    def my_mse(self, y_true, y_pred):
        # custom loss function for model
        nonzero = tf.to_float(tf.not_equal(y_true, tf.constant(0, 
            dtype=tf.float32)), name='ToFloat')
        return tf.reduce_mean(tf.multiply(tf.square(tf.subtract(y_true, 
            y_pred)), nonzero))


    def generate_floats(self):
        # generator to get training data for model
        users = np.zeros((self.block_size, self.num_users), dtype=np.float32)
        while True:
            for u in range(0, self.num_users, self.block_size):
                if u + self.block_size >= self.num_users:
                    break
                user_vecs = users.copy()
                user_vecs[np.arange(0, self.block_size), np.arange(u, u + self.block_size)] = 1
                movie_vecs = self.base[u:u + self.block_size].toarray()
                yield user_vecs, movie_vecs


    def make_model(self):
        self.rmse = real_RMSE(self.save_model, self.save_epochs)
        self.model = Sequential()
        self.model.add(Dense(self.K, input_shape=(self.num_users,), activation='linear'))  # hidden layer
        self.model.add(Dense(self.num_movies, activation='linear'))  # output layer
        self.model.summary()  # double-check model format
        # 'sgd' optimizer might not be a bad idea instead of adam:
        self.model.compile(loss=self.my_mse, optimizer=optimizers.adam())       


    def train(self):
        self.make_model()
        self.model.fit_generator(self.generate_floats(), samples_per_epoch=self.num_users,
            nb_epoch=self.epochs, verbose=True, callbacks=[self.rmse])
        # note that rmse will be sqrt(my_mse / sparsity) = srqt(my_mse * 80)


    def save_model(self):
        self.save_name = ('nnmf-k' + str(self.K) + '-e' + str(self.epochs) 
            + '-rmse' + str(self.rmse.losses[-1])[:5])
        self.model.save_weights('../models/' + self.save_name + '.h5')
        print('model saved as', self.save_name)


    def predict(self):
        print('predicting from model...')
        qual = read_arr('qual')
        print('maximums', np.max(qual, axis=0))
        qual[:, :2] -= 1
        qual_ratings = []
        user = -1
        print('maximums', np.max(qual, axis=0))
        userv = np.zeros((1, self.num_users), dtype=np.float32)
        for p, point in enumerate(qual):
            if point[0] != user:
                if (point[0] % 200 == 0):
                    print('percent done:', (100 * p / len(qual)))
                user = point[0]
                user_vec = userv.copy()
                user_vec[0, user] = 1
                movie_vec = self.model.predict(user_vec)[0] + self.data_mean
            qual_ratings.append(movie_vec[point[1]])
        qual_ratings = np.array(qual_ratings, dtype=np.float32)
        print(qual_ratings.shape)

        # save predictions
        np.savetxt('../data/submissions/' + self.save_name + '.dta', qual_ratings, fmt='%.3f', newline='\n')
        print('finished!')



# metric callback class to print custom RMSE for model
class real_RMSE(keras.callbacks.Callback):
    def __init__(self, save, save_epochs):
        '''
        Uses arg:generate_testx function to generation test validation 
        data. Uses arg:train_rmse function to generating training 
        error (on base), which is now defunct (its advantage is 
        generating RMSE after each epoch whereas keras average RMSE during 
        each epoch).
        '''
        keras.callbacks.Callback.__init__(self)
        self.save = save
        self.save_epochs = save_epochs

    def on_train_begin(self, logs={}):
        self.sparsity = 86.4229420728  # sparsity of data in full matrix
        self.e = 0
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        rmse = np.sqrt(self.sparsity * logs.get('loss'))
        print('\nReal RMSE:', rmse)
        self.losses.append(rmse)
        self.e += 1
        if self.save_epochs:
            if self.save_epochs == 'all':
                self.save()
            elif self.e in self.save_epochs:
                self.save()



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