from cache import *
from sklearn.decomposition import TruncatedSVD
import time
start = time.time()

# define K components and number of iterations for SVD
K, iters = 30, 1

# read in data and perform SVD
data = sp_mat_read()
svd = TruncatedSVD(n_components=K, n_iter=iters)
svd.fit(data)
print('explained variance:', svd.explained_variance_ratio_.sum())

# make predictions for data in qual
print('making predictions on qual based on factorized svd...')
qual = np_read(name='qual')
qual_ratings = []
for p, point in enumerate(qual):
    qual_ratings.append()
qual_ratings = np.array(qual_ratings, dtype=np.float32)

# save predictions
np.savetxt('../data/qual_svd_%d_%d_%d.dta' % (K, iters, int(time.time() - start)), 
    qual_ratings, fmt='%.3f', newline='\n')
print('finished!')



'''
In-sample:
K = 50
(i=1, .3485) (i=5, .363)
K = 30
'''