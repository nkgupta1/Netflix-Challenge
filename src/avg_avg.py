#!/usr/bin/env python3
import numpy as np 
import pandas as pd 

# #calculate average user rating across users 
# def avg_user(all, test): 
# 	[nrows, ncols] = test.shape
# 	pred = np.empty((nrows))
# 	group_user = all.groupby('user')[['rating']]
# 	avg_user = group_user.aggregate(np.mean)

# 	for k in range(nrows): 
# 		user = test.at[k,'user']
# 		pred[k] = np.mean(avg_user.loc[user])

# 	return(pred)

# #calculate average movie rating across movies 
# def avg_movie(all, test):
# 	[nrows, ncols] = test.shape
# 	pred = np.empty((nrows))
# 	group_movie = all.groupby('movie')[['rating']]
# 	avg_movie = group_movie.aggregate(np.mean)

# 	for k in range(nrows): 
# 		movie = test.at[k,'movie']
# 		pred[k] = np.mean(avg_movie.loc[movie])

# 	return(pred)

# #calculate average of average user rating and movie rating 
# #for each (user, movie) pair 
# def avg_avg(all, test):

# 	pred_user = avg_user(all, test)
# 	pred_movie = avg_movie(all, test)
# 	pred_avg = (pred_user+pred_movie)/2 

# 	return pred_avg 

#compute predictions 
# base = pd.read_csv("../data/um/base_all.dta", delim_whitespace=True, header=None, 
# 	names=['user', 'movie', 'time', 'rating'])
# test = pd.read_csv("../data/um/qual_all.dta", delim_whitespace=True, header=None, 
# 	names=['user', 'movie', 'time'])

user_averages = np.zeros((458293,2))
movie_averages = np.zeros((17770,2))

g = open("../data/um/base_all.dta",'r')

i = 0
for line in g:
	if (((i+1) % 25000000) == 0):
		print('.')
	elif (((i+1) % 1000000) == 0):
		print('.', end='', flush=True)
	i += 1

	lst = line.split()
	u = int(lst[0])
	m = int(lst[1])
	# Ignore date
	r = int(lst[3])

	if (user_averages[u-1,0] == 0):
		user_averages[u-1,0] = r
		user_averages[u-1,1] = 1
	else:
		user_averages[u-1,0] *= user_averages[u-1,1] # Scale add
		user_averages[u-1,0] += r # Add
		user_averages[u-1,1] += 1 # Scale up base
		user_averages[u-1,0] /= user_averages[u-1,1] # Scale down

	if (movie_averages[m-1,0] == 0):
		movie_averages[m-1,0] = r
		movie_averages[m-1,1] = 1
	else:
		movie_averages[m-1,0] *= movie_averages[m-1,1] # Scale add
		movie_averages[m-1,0] += r # Add
		movie_averages[m-1,1] += 1 # Scale up base
		movie_averages[m-1,0] /= movie_averages[m-1,1] # Scale down

# Next, re-run with base+probe for the qual stuff
f = open('avg_avg_probe.txt','w')
h = open("../data/um/probe_all.dta")
for line in h:
	lst = line.split()
	u = int(lst[0])
	m = int(lst[1])

	user_ave = user_averages[u-1,0]
	movie_ave = movie_averages[m-1,0]

	ave_ave = (user_ave + movie_ave) / 2
	f.write(str(ave_ave)+'\n')