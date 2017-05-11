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
    # print("Creating RBM...")
    # rbm = RBM(M, latent_factors, num_ratings, learning_rate)
    # rbm = rbm.load_RBM("../data/og_rbm.pkl")

    # # Now, predict samples
    # print("Beginning prediction...")
    # input_data = np.zeros((M,num_ratings))
    # with open("../data/rbm_pred_temp2.dta", 'w') as g:
    #     with open(dataset) as f:
    #         count = 1

    #         overflow_line = ""
    #         lines_to_predict = []
    #         while (count <= U):
    #             print("    Predicting user {0}/{1}".format(str(count),str(U)))

    #             # Zero out array
    #             input_data.fill(0)

    #             # Get the relevant lines
    #             if (overflow_line != ""):
    #                 olst = overflow_line.split()

    #                 rating = int(olst[3])

    #                 # input_data[movie][rating-1] = rating
    #                 if rating:
    #                     input_data[int(olst[1])-1][rating-1] = 1
    #                 else:
    #                     lines_to_predict.append(overflow_line)

    #             overflow_lines = ""

    #             while (1):
    #                 line = f.readline()
    #                 lst = line.split()

    #                 if (len(lst) != 4):
    #                     break

    #                 # Data point is for next user
    #                 if (int(lst[0]) > count):
    #                     overflow_line = line
    #                     break
    #                 else:
    #                     rating = int(lst[3])

    #                     # input_data[movie][rating-1] = rating
    #                     if rating:
    #                         input_data[int(lst[1])-1][rating-1] = 1
    #                     else:
    #                         lines_to_predict.append(line)

    #             # Do stuff with lines, including training
    #             # Note that this can create entries with entirely None
    #             # Make sure that works for RBM

    #             # Now, actually predict
    #             for line in lines_to_predict:
    #                 lst = line.split()

    #                 user = int(lst[0])
    #                 movie = int(lst[1]) - 1 # Zero-index everything
    #                 rating = int(lst[3])

    #                 if (rating == 0):
    #                     pred = rbm.get_prediction(movie, rbm.get_hidden_probabilities(input_data, getRated(input_data)))
    #                     # print("User: {0}, Movie: {1}, Rating: {2}".format(str(user), str(movie+1), str(pred)))
    #                     # g.write(str(pred) + '\n')
    #                     g.write(str(user)+','+str(movie+1)+','+str(pred)+'\n')
    #                 else:
    #                     continue       

    #             lines_to_predict = []
    #             count += 1
    # print("Prediction Completed")

    print("Beginning data cleansing")
    with open("../data/rbm_pred2.dta", 'w') as g:
        with open("../data/rbm_pred_temp2.dta") as f:
            while (1):
                line = f.readline()
                lst = line.split(',')

                if (len(lst) != 3):
                    break

                rating = float(lst[2])

                # Make sure rating not above 5, lower than 1
                new_rating = round(max(1.0, min(5.0, rating)),4)

                g.write(str(new_rating) + '\n')



    # V = [[[0,0,0,0,1],None],[[0,0,0,0,1],None],[[0,0,0,0,1],None]]
    # rbm.learn_batch(0, V)
    # pred = rbm.get_prediction(0, rbm.get_hidden_probabilities(V[0], getRated(V[0])))
    # print(pred)