import numpy as np 
import pandas as pd 

#calculate average user rating across users 
def avg_user(all, test): 
	[nrows, ncols] = test.shape
	pred = np.empty((nrows))
	group_user = all.groupby('user')[['rating']]
	avg_user = group_user.aggregate(np.mean)

	for k in range(nrows): 
		user = test.at[k,'user']
		pred[k] = np.mean(avg_user.loc[user])

	return(pred)

#calculate average movie rating across movies 
def avg_movie(all, test):
	[nrows, ncols] = test.shape
	pred = np.empty((nrows))
	group_movie = all.groupby('movie')[['rating']]
	avg_movie = group_movie.aggregate(np.mean)

	for k in range(nrows): 
		movie = test.at[k,'movie']
		pred[k] = np.mean(avg_movie.loc[movie])

	return(pred)

#calculate average of average user rating and movie rating 
#for each (user, movie) pair 
def avg_avg(all(, test):)

	pred_user = avg_user(all, test)
	pred_movie = avg_movie(all, test)
	pred_avg = (pred_user+pred_movie)/2 

	return pred_avg 

#compute predictions 
all = pd.read_csv('mu/all.dta', delim_whitespace=True, header=None, 
	names=['user', 'movie', 'time', 'rating'])
test = pd.read_csv('mu/qual.dta', delim_whitespace=True, header=None, 
	names=['user', 'movie', 'time'])
final_pred = avg_avg(all, test)

print(final_pred)
