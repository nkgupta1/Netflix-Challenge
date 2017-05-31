import io
import csv
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from functools import lru_cache
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def read_data(file):
    f = open(file)

    idxs = np.zeros(458294).astype(int) # Plus 1 for ease of use

    # Keep track of max use seen so far; use for indexing
    max_u = 0;
    for i, line in enumerate(f):
        lst = line.split(" ")
        u = int(lst[0])

        if (u > max_u):
            idxs[int(u)-1] = i;
            max_u = u;

    # Last entry is the number of lines
    idxs[-1] = i

    # Dta_ids encounters issues if users are skipped.
    # Fix those here.
    for i in range(1, 458293):
        if (idxs[i] == 0):
            idxs[i] = idxs[i-1]

    return idxs

class DataSet:

    #############
    # INTERFACE #
    #############

    """
    // How to use //

    Initialize the class with a dataset ('movielens', 'jester' or 'toy'), e.g:
    ds = DataSet(dataset='movielens')

    Once loaded, to get the dataframe with columns = [ user_id, item_id, rating ]:
    df = ds.get_df()

    If the toy dataset was chosen, one can access the full dataset:
    df_complete = ds.get_df_complete()

    Instead of the dataframe, one can get the dense rating matrix:
    dense_matrix = DataSet.df_to_matrix(df)
    
    To get some infos on the df, run:
    ds.get_description()

    To get a train / test dataframe:
    train_df, test_df = ds.split_train_test(False)

    Once the model trained, U and V built, one can get the prediction dataframe:
    pred_df = DataSet.U_V_to_df(U, V, None, test_df)

    Finally, to assess the accuracy of the model:
    score = DataSet.get_score(test_df, pred_df)
    """

    ####################
    # Static variables #
    ####################

    ## Column names
    USER_ID = 'user_id'
    ITEM_ID = 'item_id'
    RATING = 'rating'
    TIMESTAMP = 'timestamp'

    ## Dataset constants
    # DATASETS = ['movielens', 'jester', 'toy'] ## All datasets
    # DATASETS_WITH_SIZE = ['movielens']
    # DATASETS_TO_BE_FETCHED = ['movielens', 'jester']
    # SIZE = ['S', 'M', 'L']

    
    ###############
    # Constructor #
    ###############

    def __init__(self, data_file):
        """
        @Parameters:
        ------------
        dataset: String -- 'movielens' or 'jester' or 'toy'
        size:    String -- 'S', 'M' or 'L'(only for 'movielens')
        u, i, u_unique, i_unique, density, noise, score_low, score_high -- See get_df_toy (only for toy dataset)

        @Infos:
        -------
        For movielens:
            -> Size = S:   100K ratings,  1K users, 1.7K movies, ~   2MB, scale: [ 1  , 5 ], density:
            -> Size = M:     1M ratings,  6K users,   4K movies, ~  25MB, scale: [ 1  , 5 ], density: 4.26%
            -> Size = L:    10M ratings, 72K users,  10K movies, ~ 265MB, scale: [0.5 , 5 ], density: 0.52%

            All users have rated at least 20 movies no matter the size of the dataset

        For jester:
            -> Uniq. size: 1.7M ratings, 60K users,  150  jokes, ~  33MB, scale: [-10 , 10], density: 31.5%
               Values are continuous.
        """


        # Check inputs
        # if dataset not in DataSet.DATASETS:
        #     raise NameError("This dataset is not allowed.")
        # if size not in DataSet.SIZE and dataset in DataSet.DATASETS_WITH_SIZE:
        #     raise NameError("This size is not allowed.")

        # Configure parameters
        # if dataset in DataSet.DATASETS_TO_BE_FETCHED:
        # self.__set_params_online_ds(dataset)
        # # else:
        # #     self.__set_params_toy_ds(u, i, u_unique, i_unique, density, noise, score_low, score_high)

        # self.df, self.df_complete = self.__set_df()
        # self.nb_users = len(np.unique(self.df[DataSet.USER_ID]))
        # self.nb_items = len(np.unique(self.df[DataSet.ITEM_ID]))
        # self.low_user = np.min(self.df[DataSet.USER_ID])
        # self.high_user = np.max(self.df[DataSet.USER_ID])
        # self.low_rating = np.min(self.df[DataSet.RATING])
        # self.high_rating = np.max(self.df[DataSet.RATING])
        # self.item_index_range = np.max(self.df[DataSet.ITEM_ID]) - np.min(self.df[DataSet.ITEM_ID]) + 1

        # #Train and test set
        # self.df_train, self.df_test, self.df_heldout = self.split_train_test(users_size=users_size)
        # self.nb_users_train = len(np.unique(self.df_train[DataSet.USER_ID]))
        # self.nb_items_train = len(np.unique(self.df_train[DataSet.ITEM_ID]))

        self.df = pd.read_csv(data_file, sep =" ", header=None)
        self.df.columns = ['user_id', 'item_id', 'timestamp', 'rating']
        self.df = self.df[['user_id', 'item_id', 'rating']]

        self.idxs = read_data(data_file)

        self.df_complete = None
        self.nb_users = 458293
        self.nb_items = 17770
        self.low_user = 1
        self.high_user = 458293
        self.low_rating = 1
        self.high_rating = 5
        self.item_index_range = 17770

        self.df_train = self.df

    ##################
    # Public methods #
    ##################

    def get_items_user(self, user):
        """
        returns indices of items rated by this user
        """
        values = self.df[self.idxs[user-1]:self.idxs[user]]['item_id']
        return values

    def get_ratings_user(self, user):
        """
        returns observed ratings by this user
        """
        values = self.df[self.idxs[user-1]:self.idxs[user]]['rating']
        return values

    def get_users(self):
        return range(1,458293)

    def get_df(self):
        return self.df
    
    def get_description(self):
        return {
            "Number of users": self.nb_users,
            "Number of items": self.nb_items,
            "Lowest user": self.low_user,
            "Highest user": self.high_user,
            "Density": self.df.shape[0] / (self.nb_items * self.nb_users),
            "Mean of ratings": np.mean(self.df[DataSet.RATING]),
            "Standard deviation of ratings": np.std(self.df[DataSet.RATING])
        }
