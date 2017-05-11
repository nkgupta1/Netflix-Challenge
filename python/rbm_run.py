#!/usr/bin/env python3

from RBM import *
import numpy as np

U = 23
# U = 458293
M = 17770

batch_size = 10
full_iters = 1
cd_iters = 10

latent_factors = 1
num_ratings = 5 # Don't change this
learning_rate = 1 # Do feel free to change this

def onehot(rating):
    """
    Creates one-vector for a certain rating
    """
    vec = [0 for i in range(5)]
    vec[rating - 1] = 1
    return np.array(vec)

if __name__=='__main__':
    print("Creating RBM...")
    rbm = RBM(M, latent_factors, num_ratings, learning_rate)
    print("RBM created.")

    print("Beginning training...")
    # input_data = np.zeros((batch_size,M,num_ratings)).tolist()
    for d in range(full_iters):
        print("Starting iteration {0}/{1}".format(str(d+1),str(full_iters)))
        # Training phase - train with 10 users at a time
        # TODO - edit RBM to accept numpy data, in general suck less
        with open("../data/um/test.dta") as f:
            max_index = batch_size

            overflow_lines = []
            while (max_index < U + batch_size):
                print("    At user {0}/{1}".format(str(min(max_index,U)),str(U)))
                
                # input_data = [[None for j in range(M)] for i in range(batch_size)]
                input_data = np.zeros((batch_size,M,num_ratings)).tolist()

                # Read in overflow lines
                for line in overflow_lines:
                    lst = line.split()

                    user = (int(lst[0]) - 1) % batch_size
                    movie = int(lst[1]) - 1 # Zero-index everything
                    rating = int(lst[3])

                    if (rating == 0):
                        continue
                    else:
                        input_data[user][movie] = onehot(rating)    
                overflow_lines = []  

                # Get the relevant lines
                while (1):
                    line = f.readline()
                    lst = line.split()

                    if (len(lst) != 4):
                        break

                    nominal_user = int(lst[0])

                    if (nominal_user > max_index):
                        overflow_lines.append(line)
                        break

                    # Do stuff with lines, including training
                    # Note that this can create entries with entirely None
                    # Make sure that works for RBM
                    # input_data = [[None for j in range(M)] for i in range(batch_size)]
                    user = (int(lst[0]) - 1) % batch_size
                    movie = int(lst[1]) - 1 # Zero-index everything
                    rating = int(lst[3])

                    if (rating == 0):
                        continue
                    else:
                        input_data[user][movie] = onehot(rating)

                rbm.learn_batch(cd_iters,input_data)

                max_index += batch_size
    print("Training completed")

    # Now, predict samples
    print("Beginning prediction...")
    # input_data = np.zeros((M,num_ratings)).tolist()
    with open("../data/rbm_pred_temp.dta", 'w') as g:
        with open("../data/um/test.dta") as f:
            count = 1

            lines = []
            overflow_lines = []
            while (count <= U):
                print("    Predicting user {0}/{1}".format(str(count),str(U)))

                # Get the relevant lines
                while (1):
                    line = f.readline()
                    lst = line.split()

                    if (len(lst) != 4):
                        break

                    user = int(lst[0])

                    if (user > count):
                        overflow_lines.append(line)
                        break
                    else:
                        lines.append(line)

                # Do stuff with lines, including training
                # Note that this can create entries with entirely None
                # Make sure that works for RBM
                # input_data = [None for j in range(M)]
                input_data = np.zeros((M,num_ratings)).tolist()

                for line in lines:
                    lst = line.split()

                    user = (int(lst[0]) - 1) % batch_size
                    movie = int(lst[1]) - 1 # Zero-index everything
                    rating = int(lst[3])

                    if (rating == 0):
                        continue
                    else:
                        input_data[movie] = onehot(rating)

                # Now, actually predict
                for line in lines:
                    lst = line.split()

                    user = int(lst[0])
                    movie = int(lst[1]) - 1 # Zero-index everything
                    rating = int(lst[3])

                    if (rating == 0):
                        pred = rbm.get_prediction(movie, rbm.get_hidden_probabilities(input_data, getRated(input_data)))
                        # print("User: {0}, Movie: {1}, Rating: {2}".format(str(user), str(movie+1), str(pred)))
                        g.write(str(pred) + '\n')
                    else:
                        continue       

                lines = overflow_lines
                overflow_lines = []
                count += 1
    print("Prediction Completed")

    print("Beginning data cleansing")
    with open("../data/rbm_pred.dta", 'w') as g:
        with open("../data/rbm_pred_temp.dta") as f:
            while (1):
                line = f.readline()
                lst = line.split()

                if (len(lst) != 1):
                    break

                rating = float(lst[0])

                # Make sure rating not above 5, lower than 1
                new_rating = max(1.0, min(5.0, rating))

                g.write(str(new_rating) + '\n')


    # V = [[[0,0,0,0,1],None],[[0,0,0,0,1],None],[[0,0,0,0,1],None]]
    # rbm.learn_batch(0, V)
    # pred = rbm.get_prediction(0, rbm.get_hidden_probabilities(V[0], getRated(V[0])))
    # print(pred)