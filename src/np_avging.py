from cache import *

data = np_read() #[user, movie, date, rating]
data = data[data[:, 3] != 0]

# this is vectorized method but requires an insane amount of memory
# a = ((np.mgrid[:np.max(data[:, 0]), :data.shape[0]] == data[:, 0])[0] * data[:, 3]).mean(axis=1)
# use for loops instead

# get average rating for every user and movie
num_points = data.shape[0]
num_users = np.max(data[:, 0])
num_movies = np.max(data[:, 1])
user_avgs = np.zeros(num_users, dtype=np.uint32)
movie_avgs = np.zeros(num_movies, dtype=np.uint32)
user_counts = np.zeros(num_users, dtype=np.uint32)
movie_counts = np.zeros(num_movies, dtype=np.uint32)
for p, point in enumerate(data):
    if p % 100000 == 0:
        print('completed:', p / num_points)
    user_avgs[point[0] - 1] += point[3]
    movie_avgs[point[1] - 1] += point[3]
    user_counts[point[0] - 1] += 1
    movie_counts[point[1] - 1] += 1

print('averaging ratings with counts...')
# assert(np.min(user_counts) > 0)
# assert(np.min(movie_counts) > 0)
user_avgs = user_avgs.astype(np.float32)
user_avgs /= user_counts
movie_avgs = movie_avgs.astype(np.float32)
movie_avgs /= movie_counts

# make predictions for data in qual
print('making predictions on qual based on averages...')
qual = np_read(name='qual')
qual_ratings = []
for p, point in enumerate(qual):
    qual_ratings.append((user_avgs[point[0] - 1] + movie_avgs[point[1] - 1]) / 2)
qual_ratings = np.array(qual_ratings, dtype=np.float32)

# save predictions
np.savetxt('../data/qual_npavg.dta', qual_ratings, fmt='%.3f', newline='\n')
print('finished!')

