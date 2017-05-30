#!/usr/bin/env python3

"""
This class implements Non-linear Matrix Factorization with Gaussian Processes (NLMFGP), described in:
    Lawrence, Neil D., and Raquel Urtasun.
    "Non-linear matrix factorization with Gaussian processes."
    Proceedings of the 26th Annual International Conference on Machine Learning. ACM, 2009.


Code borrowed and modified for Breakfast Club 3.
"""

import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
import time
import pickle

# Add parent directory to python path
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from data_fetching.data_set import DataSet


class GpMf():
    def __init__(self, latent_dim, nb_data):
        self.latent_dim = latent_dim
        self.nb_data = nb_data
        self.X = np.random.normal(0, 1e-3, (nb_data, latent_dim))
        self.lin_variance = 1.0
        self.bias_variance = 0.11
        self.white_variance = 5.0
        self.y = None
        self.rated_items = None

    def log_likelihood(self):
        """return the log likelihood of the model"""
        Cj_invy, logDetC = self.invert_covariance()
        yj = np.asmatrix(self.y).T
        Nj = len(self.rated_items)
        likelihood = - 0.5 * (Nj * np.log(2 * math.pi) + logDetC + yj.T.dot(Cj_invy))
        return float(likelihood)

    def invert_covariance(self, gradient=False, nonlinear =False, kernel=linear_kernel):
        q = self.latent_dim
        Nj = len(self.rated_items)
        Xj = np.asmatrix(self.X[self.rated_items, :])
        yj = np.asmatrix(self.y).T
        s_n = self.white_variance
        s_w = self.lin_variance
        s_b = self.bias_variance
        sigNoise = s_w / s_n

        if Nj > q and not nonlinear: # we use the matrix inversion lemma
            XTX = Xj.T * Xj
            B = np.eye(q) + sigNoise * XTX
            Binv = np.linalg.pinv(B)
            _, logdetB = np.linalg.slogdet(B)
            if gradient:
                AinvX = (Xj - sigNoise * Xj * (Binv * XTX)) / s_n
                AinvTr = (Nj - sigNoise * (np.multiply(Xj * Binv, Xj)).sum()) / s_n
            Ainvy = (yj - sigNoise * Xj * (Binv * (Xj.T * yj))) / s_n
            sumAinv = (np.ones((Nj, 1)) - sigNoise * Xj * (Binv * Xj.sum(axis=0).T)) / s_n  # this is Nx1
            sumAinvSum = sumAinv.sum()
            denom = 1 + s_b * sumAinvSum
            fact = s_b / denom
            if gradient:
                CinvX = AinvX - fact * sumAinv * (sumAinv.T * Xj)
                CinvSum = sumAinv - fact * sumAinv * sumAinvSum
                CinvTr = AinvTr - fact * sumAinv.T * sumAinv

            Cinvy = Ainvy - fact * sumAinv * float(sumAinv.T * yj)
            if not gradient:
                logdetA = Nj * np.log(s_n) + logdetB
                logdetC = logdetA + np.log(denom)

        else :
            C = s_w * kernel(Xj, Xj)
            C = C + s_b + s_n * np.eye(Nj)
            Cinv = np.linalg.pinv(C)
            Cinvy = Cinv * yj
            if gradient:
                CinvX = Cinv * Xj
                CinvTr = np.trace(Cinv)
                CinvSum = Cinv.sum(axis=1)
            else:
                _, logdetC = np.linalg.slogdet(C)

        if gradient:
            return Cinvy, CinvSum, CinvX, CinvTr
        else:
            return Cinvy, logdetC

    def log_likelihood_grad(self, ):
        """Computes the gradient of the log likelihood"""
        s_w = self.lin_variance
        s_b = self.bias_variance
        s_n = self.white_variance

        yj = np.asmatrix(self.y).T
        Xj = np.asmatrix(self.X[self.rated_items, :])

        Cinvy, CinvSum, CinvX, CinvTr = self.invert_covariance(gradient=True)
        covGradX = 0.5 * (Cinvy * (Cinvy.T * Xj) - CinvX)
        gX = s_w * 2.0 * covGradX
        gsigma_w = np.multiply(covGradX, Xj).sum()
        CinvySum = Cinvy.sum()
        CinvSumSum = CinvSum.sum()
        gsigma_b = 0.5 * (CinvySum * CinvySum - CinvSumSum)
        gsigma_n = 0.5 * (Cinvy.T * Cinvy - CinvTr)
        return gX, float(gsigma_w), float(gsigma_b), float(gsigma_n)

    def objective(self):
        return -self.log_likelihood()


def fit(dataset, model, nb_iter=20, seed=42, momentum=0.9):
    data = dataset.get_df()
    param_init = np.zeros((1, 3))
    X_init = np.zeros(model.X.shape)
    prev_err = 10.0
    for iter in range(nb_iter):
        print("iteration", iter)
        tic = time.time()
        np.random.seed(seed=seed)
        state = np.random.get_state()
        users = np.random.permutation(dataset.get_users())
        count = 0
        for user in users:
            if (((count+1) % 50000) == 0):
                print('.')
            elif (((count+1) % 1000) == 0):
                print('.', end='', flush=True)

            lr = 1e-4
            y = dataset.get_ratings_user(user)
            rated_items = dataset.get_items_user(user) - 1
            model.y = y
            model.rated_items = rated_items
            grad_X, grad_w, grad_b, grad_n = model.log_likelihood_grad()
            gradient_param = np.array([grad_w * model.lin_variance,
                               grad_b * model.bias_variance,
                               grad_n * model.white_variance])
            param = np.log(np.array([model.lin_variance,
                                     model.bias_variance,
                                     model.white_variance]))
            # update X
            X = X_init[rated_items, :]
            ar = lr * 10
            X = X * momentum + grad_X * ar
            X_init[rated_items, :] = X
            model.X[rated_items, :] = model.X[rated_items, :] + X

            # update variances
            param_init = param_init * momentum + gradient_param * lr
            param = param + param_init
            model.lin_variance = math.exp(param[0, 0])
            model.bias_variance = math.exp(param[0, 1])
            model.white_variance = math.exp(param[0, 2])
            count += 1

        # Check validation error, save model
        val_err = 0.0
        for i in range(10000):
            true = vf.loc[i]['rating']
            pred = float(predict(vf.loc[i]['user_id'],vf.loc[i]['item_id']-1,model,dataset))
            val_err += math.pow((true - pred), 2)
        # And in-sample error
        in_err = 0.0
        for i in range(10000):
            true = data.loc[i]['rating']
            pred = float(predict(data.loc[i]['user_id'],data.loc[i]['item_id']-1,model,dataset))
            in_err += math.pow((true - pred), 2)

        in_err = math.sqrt(in_err / 10000.)
        val_err = math.sqrt(val_err / 10000.)
        print("end iteration", iter,  "=========================")
        print("duration iteration", time.time() - tic)
        print("In-Sample Error: ", in_err)
        print("Validation Error: ", val_err)

        if (val_err > prev_err):
            print("Model not improving. Terminating.")
        else:
            pickle.dump(model, open("nonlin_gp_model.pkl", "wb"))
            prev_err = val_err

    return model


def predict(user, test_items, model, dataset):
    y = dataset.get_ratings_user(user)
    rated_items = dataset.get_items_user(user) - 1
    model.rated_items = rated_items
    model.y = y
    X_test = np.asmatrix(model.X[test_items, :])
    X = np.asmatrix(model.X[model.rated_items, :])
    Cinvy, CinvSum, CinvX, CinvTr = model.invert_covariance(gradient=True)
    mean = model.lin_variance* X_test*(X.T*Cinvy) + Cinvy.sum() * model.bias_variance
    return mean


def perf_weak(dataset, base_dim=50):
    # print('Data set fetched')
    # print("Dataset desctiption", dataset.get_description())

    # model_init = GpMf(latent_dim=base_dim, nb_data=dataset.item_index_range)
    # print('Fit the model...')
    # model = fit(dataset=dataset, model=model_init)
    # print('Model fitted')

    # If we already have a fitted model, comment out everything above
    model_final = pickle.load(open("nonlin_gp_model.pkl", "rb"))

    # If you want to train more after loading a model, uncomment the following:
    # model_final = fit(dataset=dataset, model=model_final)


    probe_length = 1374739
    qual_length = 2749898

    probe_preds = np.zeros(probe_length)
    qual_preds = np.zeros(qual_length)

    for i in range(probe_length):
        pred = predict(vf.loc[i]['user_id'],vf.loc[i]['item_id']-1,model_final,dataset)
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1

        probe_preds[i] = pred

    for i in range(qual_length):
        pred = predict(qf.loc[i]['user_id'],qf.loc[i]['item_id']-1,model_final,dataset)
        if pred > 5:
            pred = 5
        if pred < 1:
            pred = 1

        qual_preds[i] = pred

    np.savetxt('cf_probe_preds.dta', probe_preds, fmt='%.3f', newline='\n')
    np.savetxt('cf_qual_preds.dta', qual_preds, fmt='%.3f', newline='\n')

    return

###############################################################################
# Script
#
import pandas as pd

data_file = "../../../../data/um/base_all.dta"
valid_file = "../../../../data/um/probe_all.dta"
qual_file = "../../../../data/um/qual_all.dta"

vf = pd.read_csv(valid_file, sep =" ", header=None)
vf.columns = ['user_id', 'item_id', 'timestamp', 'rating']
vf = vf[['user_id', 'item_id', 'rating']]

qf = pd.read_csv(qual_file, sep =" ", header=None)
qf.columns = ['user_id', 'item_id', 'timestamp', 'rating']
qf = qf[['user_id', 'item_id']]

if __name__ == "__main__":
    print('START')
    # MovieLens dataset 100k
    # dataset=DataSet(dataset="movielens")
    perf_weak(dataset=DataSet(data_file), base_dim=30)
    # MovieLens dataset 1M
    #perf_weak(dataset=DataSet(dataset="movielens", size="M"))
    # Toy dataset
    #perf_weak(dataset=DataSet(dataset="toy"))
    # Jester dataset
    #perf_weak(dataset=DataSet(dataset="jester"))
    print('END')
