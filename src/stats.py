#!/usr/bin/env python3
# script to read data and predictions and generate/print statistics
import numpy as np
from cache import *


def read_npavg(file='../data/qual_npavg.npy'):
    data = np.load(file)
    ratings = data[:, 3]
    # print statistics
    print('avg. of ratings', np.mean(ratings))
    print('std. of ratings', np.std(ratings))
    return data
    