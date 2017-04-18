from cache import *
from sklearn.decomposition import TruncatedSVD
import time
start = time.time()

# define K components and number of iterations for SVD
K, iters = 10, 1

# read in data and perform SVD
data = sp_mat_read()
svd = TruncatedSVD(n_components=K, n_iter=iters)
users_K = svd.fit_transform(data)
print('explained variance (sum):', svd.explained_variance_ratio_.sum())
print('explained variance (mean):', svd.explained_variance_ratio_.mean())
K_movies = svd.components_
# testing: shows svd is not ignoring zeros:
#for point in data[100000:110000].toarray():
#    print('rating:', np.dot(users_K[point[0] - 1], K_movies[:, point[1] - 1]))
del data  # free memory
'''
svd.inverse_transform(users_K) causes memory error, and so does 
multiplying matrices manually so we access components individually  
while predicting below (this is also .96/.02 = 50 times faster)
'''

# make predictions for data in qual
print('making predictions on qual based on factorized svd...')
qual = np_read(name='qual')
qual_ratings = []
for point in qual:
    rating = np.dot(users_K[point[0] - 1], K_movies[:, point[1] - 1])
    # bound predictions sensibly
    if rating < 1.:
        rating = 1.
    elif rating > 5.:
        rating = 5.
    qual_ratings.append(rating)
qual_ratings = np.array(qual_ratings, dtype=np.float32)

# save predictions
np.savetxt('../data/qual_svd_%d_%d_%d.dta' % (K, iters, int(time.time() - start)), 
    qual_ratings, fmt='%.3f', newline='\n')
print('finished!')



'''
Performance Documentation:
(iterations, in sum, in mean, out accuracy)

K = 60
(i=1, .358, .0060)
K = 50
(i=1, .348, .0070) (i=5, .363, .0073)
K = 30
(i=1, .323, .0108)
K = 10
(i=1, .272, .0272)
'''