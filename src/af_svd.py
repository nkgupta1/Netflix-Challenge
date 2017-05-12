from cache import *

# gpu can hold at most 4.0*10^8 32-bit floats

M = 458293  # number of users
N = 17770  # number of movies

data = read_arr('valid')
train = data[:10000]
train[:, :2] -= 1
test = data[5000:6000]
test[:, :2] -= 1

def SGD_Factorization(K, epochs=1000, learn_rate=0.01, regularization=0):
    # randomly initialize U and V matrices
    U = np.random.rand(K, M) - 0.5
    V = np.random.rand(K, N) - 0.5
    in_errors = []
    for ii in range(0, epochs):
        in_err = 0.
        out_err = 0.
        for [user, movie, _, rating] in train:
            rating = rating - 3.60
            # update u
            U[:, user] += learn_rate * V[:, movie] * (rating - np.dot(U[:, user], V[:, movie]))
            if regularization != 0:
                U[:, user] -= learn_rate * regularization * U[:, user]
            # update v
            V[:, movie] += learn_rate * U[:, user] * (rating - np.dot(U[:, user], V[:, movie]))
            if regularization != 0:
                V[:, movie] -= learn_rate * regularization * V[:, movie]
            # calculate in-sample error
            in_err += (rating - np.dot(U[:, user], V[:, movie])) ** 2
        for [user, movie, _, rating] in test:
            out_err += (rating - np.dot(U[:, user], V[:, movie])) ** 2
        in_errors.append(in_err)
        print 'epoch:', ii, in_err, out_err, np.mean(np.abs(np.matmul(U.T[:1000], V[:, :1000])))
    
    return in_errors


SGD_Factorization(5)
