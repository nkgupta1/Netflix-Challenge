from os import listdir
import tensorflow as tf
import numpy as np
import time
from sklearn.utils import shuffle
from cache import *
import keras
import pandas
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout
from keras import regularizers
from keras.utils.np_utils import to_categorical


class NN:
    def __init__(self, mode='bag', folder='svd3', biases=True, epochs=2, 
        layers=(300, 1200, 300), dropouts=(0.2, 0.2, 0.2), valid_per=0, 
        save_epochs=[], w_reg=0.0005, b_reg=0.0005, ratings_vector=False, 
        bag_dir='blend2', bags=128):
        '''
        Uses the svd latent factors in the folder data/arg:folder to train a 
        neural network along with indices and ratings in base (uses biases if 
        arg:biases). Creates the neural network using arg:epochs and arg:layers. 
        If arg:mode is 'train', the network is trained and saved. Else if 
        arg:mode is 'predict', the model is loaded with the name hard-coded 
        below and used to make and save a submission. If arg:mode is 'both', 
        both of these are done. If argmode is 'svd', it generates the training 
        error using the dot product of the latent factors (useful to compare 
        with network). When training, the period (epochs) between testing 
        validation (which takes time) per epoch is given by arg:valid_per.
        '''
        # (8096, 2048, 512) dropouts=(0.9, 0.9, 0.9) 
        self.data_mean = 3.60860891887339

        # weight and bias regularization
        self.w_reg, self.b_reg = w_reg, b_reg
        self.epochs, self.layers, self.dropouts = epochs, layers, dropouts
        self.folder, self.biases = folder, biases

        # if the ratings are trained as a hot vector, then we do not center 
        # the predictions around zero no include biases
        self.ratings_vector = ratings_vector
        if self.ratings_vector:
            self.biases = False
            self.data_mean = 0.

        # number of blocks to split the data into b/c of memory limitations:
        # provides much better performance than swapping >20gb in/out while 
        # training. the optimal number is dependent on individual hardware
        self.num_blocks = 1000
        
        # name of dataset to use for validation data
        self.valid_data = 'probe'

        # time for training is about 80-90 seconds/epoch
        # time for validation testing is about 35-40 seconds/epoch
        self.valid_per = valid_per

        # epochs at which to save model (excluding final epoch)
        self.save_epochs = set(save_epochs)

        predict = 'c-nnsvd-k50-e3-layers(1024, 256, 64)-dropouts(0.4, 0.4, 0.4)-regs0.0005,0.0005-rmse0.947'
        if mode == 'train':
            self.read_data()
            self.train()
            self.save_model()
        elif mode == 'predict':
            self.read_data(read_training=False)
            self.load_model('../models/' + predict + '.h5')
            self.predict()
        elif mode == 'both':
            self.read_data()
            self.train()
            self.save_model()
            self.predict()
        elif mode == 'svd':
            self.read_data()
            self.get_svd_rmse()
        elif mode == 'bag':
            self.read_data()
            for b in range(1, bags + 1):
                self.model = None
                self.train()
                self.predict(save_name=(bag_dir + '/' + str(b)))


    def load_model(self, name):
        '''
        Create the keras model according to NN's input parameter and loads
        keras model with arg:name. If the parameters of both models do not 
        match, this will fail.
        '''
        self.make_model()
        self.model_name = name.split('/')[-1]
        self.model.load_weights(name)


    def get_svd_rmse(self):
        '''
        Uses the svd latent factors and computes the RMSE on the training 
        data 'base'. Prints the RMSE of each block individually (b/c all 
        takes a long time) and then finally prints the average RMSE over all 
        the training data.
        This is only used to compare with a neural network.
        '''
        print('getting svd training rmse...')
        rmse_avg = []
        for block in range(0, self.num_blocks):
            ij = self.blocks_ij[block]
            # svd assumes dot product of u and v vectors to calculate rating
            prediction = np.sum(self.u[ij[:, 0]] * self.v[ij[:, 1]], axis=1)
            ratings = self.blocks_ratings[block][:, 0]
            rmse = np.sqrt(np.mean((prediction - ratings) ** 2))
            print('rmse', rmse)
            rmse_avg.append(rmse)
        print('avg rmse', np.mean(rmse_avg))


    def get_training_rmse(self):
        '''
        Uses the network in self.model and computes the RMSE on the training 
        data 'base'. Prints the RMSE of each block individually (b/c all 
        takes a long time) and then finally prints the average RMSE over all 
        the training data.
        '''
        print('getting model training rmse...')
        rmse_avg = []
        for block in range(0, self.num_blocks):
            ij = self.blocks_ij[block]
            # make one long vector of u and v latent factors for training
            trainx = np.concatenate((self.u[ij[:, 0]], self.v[ij[:, 1]]), axis=1)
            prediction = self.model.predict(trainx)[:, 0]
            ratings = self.blocks_ratings[block][:, 0]
            rmse = np.sqrt(np.mean((prediction - ratings) ** 2))
            print('rmse', rmse)
            rmse_avg.append(rmse)
        print('avg rmse', np.mean(rmse_avg))


    def read_data(self, read_training=True):
        '''
        Read in the latent factors and biases to use for training. Searches 
        for the files to be read in the directory self.folder. It aborts 
        if it fails to do so.
        Saves them as self.u, self.v, self.a, self.b.
        Also extracts the number of components and saves it as self.K.
        If arg:read_training, read in the training points as well.
        '''
        print('reading svd data...')
        folder = '../data/' + self.folder
        files = listdir(folder)
        try:
            names = [f[:-len(f.split('.')[-1]) - 1] for f in files]
            u = [i for i, f in enumerate(names) if f[-1] == 'u'][0]
            v = [i for i, f in enumerate(names) if f[-1] == 'v'][0]
            if self.biases:
                a = [i for i, f in enumerate(names) if f[-1] == 'a'][0]
                b = [i for i, f in enumerate(names) if f[-1] == 'b'][0]
        except:
            print('The appropriate files could not be found. Aborting.')
            quit()

        self.u = pandas.read_csv(folder + '/' + files[u], header=None, 
            sep=' ', dtype=np.float32).values[:, :-1]
        self.v = pandas.read_csv(folder + '/' + files[v], header=None, 
            sep=' ', dtype=np.float32).values[:, :-1]
        if self.biases:
            self.a = pandas.read_csv(folder + '/' + files[a], header=None, 
                sep=' ', dtype=np.float32).values[0]
            self.b = pandas.read_csv(folder + '/' + files[b], header=None, 
                sep=' ', dtype=np.float32).values[0]
            print('a=', self.a.shape, 'b=', self.b.shape)
        self.K = self.u.shape[1]
        print('k=%d' % self.K, 'u=', self.u.shape, self.u.dtype, 'v=', 
            self.v.dtype, self.v.shape)

        if read_training:
            self.read_training()


    def read_training(self):
        '''
        Read in the training data (base) and process it. Extract the 
        indices of user and movies (ij) and the ratings, then split 
        them into self.num_blocks for generation during training.
        Save as self.blocks_ij and self.blocks_ratings. The size of 
        each block is stored in self.block_size.
        '''
        print('reading training data...')
        all_data = read_arr('base')
        self.num_points = all_data.shape[0]

        print('preprocessing data...')
        original_ij, original_ratings = self.preprocess_data(all_data)
        del all_data

        print('splitting data...')
        self.num_samples = self.num_points - (self.num_points % self.num_blocks)
        self.block_size = (self.num_points // self.num_blocks)
        self.blocks_ij = self.split_data(original_ij)
        self.blocks_ratings = self.split_data(original_ratings)
        self.num_blocks += 1  # because of remainder added


    def preprocess_data(self, data):
        '''
        Preprocess the data by converting the read-in data points into 
        float ratings subtracted by the data mean and a, b vectors if 
        self.biases as well as ij indices of points subtracted by 1 for 
        zero-indexing.
        '''
        ij = (data[:, :2] - 1)  # make the data zero indexed
        if data.shape[1] == 4:
            ratings = (data[:, 3].astype(np.float32) - self.data_mean)
            if self.biases:
                ratings -= self.a[ij[:, 0]]
                ratings -= self.b[ij[:, 1]]
            ratings = np.expand_dims(ratings, axis=1)
        else:
            ratings = None  # true when qual data is read
        return ij, ratings


    def split_data(self, arr):
        '''
        Split data in numpy array arg:arr along the first axis into 
        self.num_blocks. Include the remainder of the split division 
        as the last block. Return split array.
        '''
        blocks = arr[:self.num_samples]
        remainder = arr[self.num_samples:]
        blocks = [blocks[i * self.block_size:(i + 1) * self.block_size] 
                    for i in range(0, self.num_blocks)]
        blocks.append(remainder)
        return blocks


    def generate_block(self):
        '''
        Generates training data (x, y) for the model using the indices 
        and ratings stored in self.blocks_ij and self.blocks_ratings. 
        Concatenates the u,v vectors matching the indices as x and uses 
        the ratings as y.
        '''
        block = 0
        while True:
            ij = self.blocks_ij[block]
            trainx = np.concatenate((self.u[ij[:, 0]], self.v[ij[:, 1]]), axis=1)
            ratings = self.blocks_ratings[block]
            if self.ratings_vector:
                ratings = to_categorical(ratings - 1, 5)
            yield trainx, ratings
            block = (block + 1) % self.num_blocks
            if block == 0:
                self.blocks_ij, self.blocks_ratings = shuffle(self.blocks_ij, 
                    self.blocks_ratings)


    def make_model(self):
        '''
        Make the keras model ready for predicting according to input 
        parameters. The hard-coded properties are the nodes' activation 
        types and the optimizer.
        '''
        self.model = Sequential()
        # hidden layer 1
        self.model.add(Dense(self.layers[0], input_shape=(self.K * 2,), 
            activation='linear', W_regularizer=regularizers.l2(self.w_reg),
                b_regularizer=regularizers.l1(self.b_reg)))
        if self.dropouts[0]:
            self.model.add(Dropout(self.dropouts[0]))
        # hidden layers 2, 3
        for l in [1, 2]: 
            if self.layers[l]:
                self.model.add(Dense(self.layers[l], activation='relu', 
                W_regularizer=regularizers.l2(self.w_reg),
                b_regularizer=regularizers.l1(self.b_reg)))
                if self.dropouts[l]:
                    self.model.add(Dropout(self.dropouts[l]))
        # output layer
        if self.ratings_vector:
            self.model.add(Dense(5, activation='relu'))
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam())
        else:
            self.model.add(Dense(1, activation='relu'))
            # 'sgd' optimizer might not be a bad idea instead of adam:
            self.model.compile(loss='mse', optimizer=optimizers.adam()) #lr=0.001     
        self.model.summary()  # double-check model format
        


    def train(self):
        '''
        Make and train the keras model using a generator for the training 
        data. Use custom callback self.rmse while training to print 
        training and validation RMSE. Prints the history of lossses when done.
        '''
        self.make_model()
        self.rmse = RMSE(self.generate_validation, self.get_training_rmse, 
            self.save_model, self.save_epochs, self.valid_per, self.ratings_vector)
        self.model.fit_generator(self.generate_block(), 
            samples_per_epoch=self.num_points, nb_epoch=self.epochs, 
            verbose=True, callbacks=[self.rmse])
        print('RMSE training losses:', self.rmse.losses)
        if self.rmse.validate:
            print('RMSE validation losses:', self.rmse.validation_losses)


    def save_model(self, mid_epoch=False):
        '''
        Save the weights of the model self.model in ../models/ and training 
        information to the log nn_svd_c_log.txt.
        '''
        if mid_epoch:
            epochs = mid_epoch
        else:
            epochs = self.epochs
        if self.ratings_vector:
            hot = 'hot-'
        else:
            hot = ''
        self.model_name = (hot + 'c-nnsvd-k' + str(self.K) + '-e' + str(epochs) 
            + '-layers' + str(self.layers) + '-dropouts' + str(self.dropouts) 
            + '-regs' + str(self.w_reg) + ',' + str(self.b_reg)
            + '-rmse' + str(self.rmse.losses[-1])[:5] + '.h5')
        filename = ('../models/' + self.model_name)
        self.model.save_weights(filename)
        print('model saved as:', filename)
        with open('nn_svd_c_log.txt', 'a') as log:
            log.write('\nSVD_Name=' + self.folder + ', K=' + str(self.K) +
            ', Epochs=' + str(epochs) + ', Layers=' + str(self.layers) + 
            ', Dropouts=' + str(self.dropouts) + ', Validation=' + 
            self.valid_data + ', Regularization (weights, bias)=' + 
            str((self.w_reg, self.b_reg)) + ', Hot=' + str(self.ratings_vector) 
            + '\nModel Saved: ' + filename + 
            '\nRMSE training losses: ' + 
            str(self.rmse.losses) + '\nRMSE validation losses: ' + 
            str(self.rmse.validation_losses) + '\n')

            if self.save_epochs:
                log.write('Intermediate Epochs Saved: ' + 
                    str(self.save_epochs) + '\n')


    def predict(self, save_name=None):
        '''
        Use self.model to predict the ratings for qual data, which are 
        used for submission and saved in ../data/submissions as a .dta 
        file with the name of the model.
        '''
        print('predicting from model...')
        print('preprocessing data...')
        qual_ij, _ = self.preprocess_data(read_arr('qual'))
        testx = np.concatenate((self.u[qual_ij[:, 0]], self.v[qual_ij[:, 1]]), axis=1)

        print('making predictions...', testx.shape)
        qual_ratings = self.model.predict(testx)
        if self.ratings_vector:
            qual_ratings = np.sum(qual_ratings * np.arange(1, 6), axis=1)
        else:
            qual_ratings = qual_ratings[:, 0]
        # undo pre-processing with means using self.data_mean, self.a, self.b
        print('adjusting predictions...', qual_ratings.shape)
        qual_ratings += self.data_mean
        if self.biases:
            qual_ratings += self.a[qual_ij[:, 0]] + self.b[qual_ij[:, 1]]

        print('saving predictions...', qual_ratings.shape)
        if not save_name:
            save_name = self.model_name
        np.savetxt('../data/submissions/' + save_name + '.dta', 
            qual_ratings, fmt='%.3f', newline='\n')
        print('finished!')


    def generate_validation(self):
        '''
        Used by the RMSE callback to generate the testing data (x, y)
        for printing validation error at the start of training, instead 
        of every epoch.
        '''
        print('preparing validation data for callback...')
        ij, valid_ratings = self.preprocess_data(read_arr(self.valid_data))
        testx = np.concatenate((self.u[ij[:, 0]], self.v[ij[:, 1]]), axis=1)
        valid_ratings = valid_ratings[:, 0]
        return testx, valid_ratings



class RMSE(keras.callbacks.Callback):
    '''
    Metric callback class to print custom RMSE for keras model. Passed 
    into the model when fit is called. It records the training and validation 
    RMSE in the fields self.losses and self.valid_ratings. It always 
    records training RMSE. It records validation RMSE every self.validate 
    epochs given by arg:validate of init(). If self.validate is zero, it 
    does not record validation error.
    '''
    def __init__(self, generate_testx, train_rmse, save, save_epochs, 
        validate, ratings_vector):
        '''
        Uses arg:generate_testx function to generation test validation 
        data. Uses arg:train_rmse function to generating training 
        error (on base), which is now defunct (its advantage is 
        generating RMSE after each epoch whereas keras average RMSE during 
        each epoch).
        '''
        keras.callbacks.Callback.__init__(self)
        # period of iterations between generation of validation errors:
        self.validate = validate
        # period of iterations between saving model using self.save()
        self.save, self.save_epochs = save, save_epochs
        self.ratings_vector = ratings_vector

        self.get_training_rmse = train_rmse
        if not validate:
            return
        # prepare for validation rmse by generating testing data
        self.testx, self.valid_ratings = generate_testx()


    def on_train_begin(self, logs={}):
        self.losses = []
        self.validation_losses = []
        self.e = 0  # curent number of epochs done


    def on_epoch_end(self, epoch, logs={}):
        '''
        Print training (in-sample) RMSE and append to history self.losses
        '''
        rmse = logs.get('loss')
        if not self.ratings_vector:
            rmse = np.sqrt(rmse)
            print('\nTraining RMSE:', rmse)
        else:
            print('')
        self.losses.append(rmse)
        if self.validate and (self.e % self.validate) == 0:
            self.get_validation_rmse()
        self.e += 1
        if self.e in self.save_epochs:
            self.save(mid_epoch=self.e)


    def get_validation_rmse(self):
        '''
        Print validation (out-of-sample) RMSE and append to history 
        self.validation_losses.
        '''
        predictions = self.model.predict(self.testx)
        if self.ratings_vector:
            predictions = np.sum(predictions * np.arange(1, 6), axis=1)
        else:
            predictions = predictions[:, 0]
        rmse = np.sqrt(np.mean((predictions - self.valid_ratings) ** 2))
        print('Validation RMSE:', rmse)
        if self.validate == 1:
            self.validation_losses.append(rmse)
        else:
            self.validation_losses.append((self.e, rmse))




if __name__ == '__main__':
    NN()


