#!/usr/bin/env python3

from RBM import *
import numpy as np

# U = 23
U = 458293
M = 17770

# batch_size = 10
batch_size = 100
full_iters = 10
cd_iters = 1

latent_factors = 50
num_ratings = 5 # Don't change this
learning_rate = 1 # Do feel free to change this

dataset = "../data/um/all.dta"

def onehot(rating):
    """
    Creates one-vector for a certain rating
    """
    vec = [0 for i in range(5)]
    vec[rating - 1] = 1
    return vec

if __name__=='__main__':
    print("Creating RBM...")
    rbm = RBM(M, latent_factors, num_ratings, learning_rate)
    print("RBM created.")

    print("Beginning training...")
    input_data = np.zeros((batch_size,M,num_ratings))
    for d in range(full_iters):
        print("Starting iteration {0}/{1}".format(str(d+1),str(full_iters)))
        # Training phase - train with 10 users at a time
        # TODO - edit RBM to accept numpy data, in general suck less
        with open(dataset) as f:
            max_index = batch_size

            overflow_line = ""
            while (max_index < U + batch_size):
                print("    At user {0}/{1}".format(str(min(max_index,U)),str(U)))

                # Zero out array
                input_data.fill(0)

                # Get the relevant lines
                if (overflow_line != ""):
                    olst = overflow_line.split()

                    user = (int(olst[0]) - 1) % batch_size
                    movie = int(olst[1]) - 1 # Zero-index everything
                    rating = int(olst[3])

                    if rating:
                        input_data[user][movie][rating-1] = 1 

                overflow_line = ""  

                while (1):
                    line = f.readline()
                    lst = line.split()

                    if (len(lst) != 4):
                        break

                    nominal_user = int(lst[0])

                    if (nominal_user > max_index):
                        overflow_line  = line
                        break
                    else:
                        user = (int(lst[0]) - 1) % batch_size
                        movie = int(lst[1]) - 1 # Zero-index everything
                        rating = int(lst[3])

                        if rating:
                            input_data[user][movie] = onehot(rating)

                # Do stuff with lines, including training
                # Note that this can create entries with entirely None
                # Make sure that works for RBM
                rbm.learn_batch(cd_iters,input_data)

                max_index += batch_size
                
                if (max_index % 1000 == 0):
                    print("Saving progress...")
                    rbm.save_RBM("../data/og_rbm.pkl")
    print("Training completed")

    # Now, predict samples
    print("Beginning prediction...")
    input_data = np.zeros((M,num_ratings))
    with open("../data/rbm_pred_temp.dta", 'w') as g:
        with open(dataset) as f:
            count = 1

            overflow_line = ""
            lines_to_predict = []
            while (count <= U):
                print("    Predicting user {0}/{1}".format(str(count),str(U)))

                # Zero out array
                input_data.fill(0)

                # Get the relevant lines
                if (overflow_line != ""):
                    olst = overflow_line.split()

                    rating = int(olst[3])

                    # input_data[movie][rating-1] = rating
                    if rating:
                        input_data[int(olst[1])-1][rating-1] = 1
                    else:
                        lines_to_predict.append(overflow_line)

                overflow_lines = ""

                while (1):
                    line = f.readline()
                    lst = line.split()

                    if (len(lst) != 4):
                        break

                    # Data point is for next user
                    if (int(lst[0]) > count):
                        overflow_line = line
                        break
                    else:
                        rating = int(lst[3])

                        # input_data[movie][rating-1] = rating
                        if rating:
                            input_data[int(lst[1])-1][rating-1] = 1
                        else:
                            lines_to_predict.append(line)

                # Do stuff with lines, including training
                # Note that this can create entries with entirely None
                # Make sure that works for RBM

                # Now, actually predict
                for line in lines_to_predict:
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

                lines_to_predict = []
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
                new_rating = round(max(1.0, min(5.0, rating)),4)

                g.write(str(new_rating) + '\n')


    rbm.save_RBM("../data/og_rbm.pkl")
    # rbm = RBM(2,50,5,1)
    # V = [[[0,0,1,0,0],None] for i in range(20)]
    # rbm.learn_batch(10, V)
    # pred = rbm.get_prediction(0, rbm.get_hidden_probabilities(V[0], getRated(V[0])))
    # print(pred)