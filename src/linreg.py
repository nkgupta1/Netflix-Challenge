#!/usr/bin/env python3
import numpy as np 
import pandas as pd 


def linreg(all):
	#linearly regress user and movies on rating, return coefficients 
	U = all[['user', 'movie']].values
	R = all[['rating']].values
	left = np.array(np.linalg.inv(np.dot(np.transpose(U), U)))
	right = np.array(np.dot(np.transpose(U), R))
	b = np.array(np.dot(left, right))

	return(b)

all = pd.read_csv('mu/all.dta', delim_whitespace=True, header=None, 
	names=['user', 'movie', 'time', 'rating'])
test = pd.read_csv('mu/qual.dta', delim_whitespace=True, header=None, 
	names=['user', 'movie', 'time'])

coef = linreg(all)
test_U = test[['user', 'movie']].values
pred = np.transpose(np.dot(test_U, coef))[0]

print(pred)
