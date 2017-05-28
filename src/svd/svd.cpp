
#include "svd.hpp"

using namespace std;


float dot_product(float **U, float *X_norm, float **X_sum, int i, float **V, int j) {
    /*
     * Calculates (U[user] + norm[user]*X_sum[user])*V[movie]
     */
    
    float dot_prod = 0.;
    for (int k = 0; k < K; k++) {
        dot_prod += (U[i][k] + X_norm[i] * X_sum[i][k]) * V[j][k];
    }
    return dot_prod;
}

float prediction(svd_data *data, int i, int j) {
    /*
     * generates a prediction based on trained matrices
     *
     * i is the user index and j is the movie index
     */
    float dot_prod = dot_product(data->U, data->X_norm, data->X_sum, i, data->V, j);
    return dot_prod + data->a[i] + data->b[j] + MEAN;
}

float get_err(svd_data *data, float **Y, int num_pts, float reg) {
    /*
     * Get the squared error of (U + X)*V^T on Y.
     */
    double err = 0.0;
    float i, j, Yij, pred;

    // calculate the mean squared error on each training point
    for (int ind = 0; ind < num_pts; ind++) {
        i    = Y[ind][0];
        j    = Y[ind][1];
        Yij  = Y[ind][3];

        pred = prediction(data, i-1, j-1);

        err += (Yij - pred) * (Yij - pred);
    }

    // if reg is not 0, then we are using this error for training so include
    // regularization
    // if (reg != 0) {
    //     err /= 2.0;
    //     err /= num_pts;

    //     double U_frob_norm = 0.0;
    //     double V_frob_norm = 0.0;
    //     double a_frob_norm = 0.0;
    //     double b_frob_norm = 0.0;

    //     for (int i = 0; i < M; i++) {
    //         for (int j = 0; j < K; j++) {
    //             U_frob_norm += data->U[i][j] * data->U[i][j];
    //         }
    //     }

    //     for (int i = 0; i < N; i++) {
    //         for (int j = 0; j < K; j++) {
    //             V_frob_norm += data->V[i][j] * data->V[i][j];
    //         }
    //     }

    //     for (int i = 0; i < M; i++) {
    //         a_frob_norm += data->a[i] * data->a[i];
    //     }

    //     for (int i = 0; i < N; i++) {
    //         b_frob_norm += data->b[i] * data->b[i];
    //     }

    //     err += 0.5 * reg * U_frob_norm;
    //     err += 0.5 * reg * V_frob_norm;
    //     err += 0.5 * reg * a_frob_norm;
    //     err += 0.5 * reg * b_frob_norm;
    // } else {
        // if reg is 0, we are using this error for testing so want RMSE error
    err /= num_pts;
    err = sqrt(err);
    // }

    return err;

}

float rand_num() {
    /*
     * returns a random number from [-0.5, 0.5]
     * note this isn't a very good random...
     */
    
    float r = static_cast <float> (rand()); 
    r = r / static_cast <float> (RAND_MAX);
    return r - 0.5;

}

svd_data* train_model(float eta, float reg, float **Y_train, float **Y_test,
                      float eps, int max_epochs) {    
    /*
     * Train an SVD model using SGD.
     *
     * Factorizes a sparse M x N matrix in the product of a M x K matrix and a N
     * x K matrix such that Y = (U + X) * V^T + a + b
     *
     * U is the user by latent factor matrix
     * X is the movie by latent factor matrix ignoring ratings
     * X_sum is the sum of all the latent factors for the movie the user rated
     * V is the movie by latent factor matrix
     * list_movies is a user x number of movies rated by that user matrix and 
     *     contains the id of all the movies that user rated
     * 
     * a is user biases
     * b is movie biases
     *
     * eta is the learning rate.
     * reg is the regularization
     * eps is the stopping criterion
     */

    ////////////////////////////////////////////////////////////////////////////

    printf("initializing matrices...\n");
    
    srand(time(NULL));

    ////////////////////////////////////////////////////////////////////////////

    float ** U = new float*[M];
    for (int i = 0; i < M; i++) {
        U[i] = new float[K];
    }

    float ** V = new float*[N];
    for (int i = 0; i < N; i++) {
        V[i] = new float[K];
    }

    float ** X = new float*[K];
    for (int i = 0; i < K; i++) {
        X[i] = new float[N];
    }

    float ** X_sum = new float*[M];
    for (int i = 0; i < M; i++) {
        X_sum[i] = new float[K];
    }

    float *X_norm = new float[M];
    float *X_temp_updates = new float[K];
    int *movie_counts = new int[M];
    float *a = new float[M];
    float *b = new float[N];

    ////////////////////////////////////////////////////////////////////////////

    // populate with random values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            U[i][j] = rand_num();
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            V[i][j] = rand_num();
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            X[i][j] = rand_num();;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            X_sum[i][j] = 0.;
        }
    }

    for (int i = 0; i < M; i++) {
        X_norm[i] = 0.;
    }

    for (int i = 0; i < M; i++) {
        movie_counts[i] = 0;
    }

    for (int i = 0; i < M; i++) {
        a[i] = rand_num();;
    }

    for (int i = 0; i < N; i++) {
        b[i] = rand_num();;
    }

    ////////////////////////////////////////////////////////////////////////////


    // init X_sum to be correct
    int user, movie;
    for (int i = 0; i < NUM_TRAIN_PTS; i++) {
        user = Y_train[i][0];
        movie = Y_train[i][1];

        // count the number of movies each user has rated
        movie_counts[user-1]++;
        for (int k = 0; k < K; k++) {
            X_sum[user-1][k] += X[k][movie-1];
        }
    }

    // i hate myself but....
    // here is a list of movies that every user has rated
    int ** list_movies = new int*[M];
    for (int i = 0; i < M; i++) {
        list_movies[i] = new int[movie_counts[i]];
    }

    // now populate this list...
    for (int i = 0; i < NUM_TRAIN_PTS;) {
        int user = Y_train[i][0] - 1;
        // loop over the movies that this user has rated
        // MUST BE IN UM ORDER!!!!!!!!!!!
        for (int j = 0; j < movie_counts[user]; j++) {
            list_movies[user][j] = Y_train[i+j][1] - 1;
        }
        i += movie_counts[user];
    }

    // now actually calculate the norm
    for (int i = 0; i < M; i++) {
        X_norm[i] = 1. / sqrt(movie_counts[i]);
    }


    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;
    toRet->X = X;
    toRet->X_sum = X_sum;
    toRet->X_norm = X_norm;
    toRet->a = a;
    toRet->b = b;

    // so we can randomly go over all the points
    int *indices = new int[NUM_TRAIN_PTS];
    for (int i = 0; i < NUM_TRAIN_PTS; i++) {
        indices[i] = i;
    }

    // takes 9 seconds
    // NOTE IF UNCOMMENTED, COULD BREAK CODE
    // random_shuffle(&indices[0], &indices[NUM_TRAIN_PTS-1]);

    ////////////////////////////////////////////////////////////////////////////

    // scales eta by this every time we have trouble training
    float adaptive_learning_rate = 0.9;

    // float delta = 0.0;
    int ind;
    int date, Yij, prev_user;
    float pred_err, old_U, old_V, norm, old_sum, update;
    // float before_E_in;
    float E_in, E_out;
    float min_E_out = 100;
    // float org_eta = eta;
    // this variable contains the number of times that E_out has consecutively
    // gone up
    int count_E_out_up = 0;

    printf("starting training....\n");

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // before_E_in = get_err(toRet, Y_train, NUM_TRAIN_PTS, 0);

        // num_grad_issues = 0;
        prev_user = -1;
        for (int i = 0; i < K; i++) {
            X_temp_updates[i] = 0;
        }


        for (int ind_ind = 0; ind_ind < NUM_TRAIN_PTS; ind_ind++) {
            ind   = indices[ind_ind];
            user  = Y_train[ind][0];
            movie = Y_train[ind][1];
            date  = Y_train[ind][2];
            Yij   = Y_train[ind][3];

            if (user - 1 != prev_user - 1 && prev_user != -1) {
                // update each movie that this user rated
                for (int k = 0; k < K; k++) {
                    for (int i = 0; i < movie_counts[prev_user-1]; i++) {
                        update = eta*(X_temp_updates[k] - reg*X[k][list_movies[prev_user-1][i]]);

                        X[k][list_movies[prev_user-1][i]] += update;
                        X_sum[prev_user-1][k] += update;
                    }

                    // now that we have updated the cumulated updates for the
                    // movies, we can reset the updates as we move onto the next
                    // user
                    X_temp_updates[k] = 0.;

                }
            }

            // so compiler doesn't generate warnings
            (void)date;
            // (void)delta;
            // (void)before_E_in;

            pred_err = Yij - prediction(toRet, user-1, movie-1);

            for (int k = 0; k < K; k++) {
                old_U = U[user-1][k];
                old_V = V[movie-1][k];
                norm = X_norm[user-1];
                old_sum = X_sum[user-1][k];

                // update user matrix
                U[user-1][k] += eta*(old_V*pred_err - reg*old_U);

                // update movie matrix
                V[movie-1][k] += eta*((old_U + norm*old_sum)*pred_err - reg*old_V);

                // update our temp update vector for X because it is too slow to
                // update it every time. won't get the same accuracy but makes
                // it much quicker...
                X_temp_updates[k] += pred_err * norm * old_V;
            }

            
            // update a
            a[user-1] += eta*(pred_err - reg*a[user-1]);

            // update b
            b[movie-1] += eta*(pred_err - reg*b[movie-1]);

        }

        // if (num_grad_issues != 0)
            // printf("num grad issues: %d\n", num_grad_issues);

        // get the error for the epoch
        E_in  = get_err(toRet, Y_train, NUM_TRAIN_PTS, 0);
        E_out = get_err(toRet, Y_test, NUM_TEST_PTS, 0);

        printf("iteration %3d:\tE_in: %1.5f\tE_out: %1.5f\tdel E_out: %1.7f\tdel E\'s: %1.7f\n", 
            epoch, E_in, E_out, min_E_out - E_out, E_out - E_in);
        
        if (E_out < min_E_out) {
            min_E_out = E_out;
            count_E_out_up = 0;
            // save_matrices(toRet, E_out, org_eta, reg, epoch);
        } else {
            count_E_out_up++;
            eta = adaptive_learning_rate * eta;
            if (count_E_out_up > 4) {
                break;
            }
        }

        // check termination condition
        // if (epoch == 0) {
        //     delta = before_E_in - E_in;
        // } else if (before_E_in - E_in < eps * delta) {
        //       break;          
        // }

    }

    printf("finished training...\n\n");


    // clean up
    delete X_temp_updates;
    // delete[] indicies;

    return toRet;

}

float **read_data(const string file, int num_pts) {
    /*
     * Read in data for training or test.
     * In format:
     *     user, movie, date, rating
     */
    
    printf("reading in data...\n");
    
    // preallocate array
    float ** Y = new float*[num_pts];
    for (int i = 0; i < num_pts; i++) {
        Y[i] = new float[4];
    }

    // read in the data
    ifstream datafile(file);
    int u, m, d, r;
    for (int i = 0; i < num_pts; i++) {
        datafile >> u >> m >> d >> r;

        Y[i][0] = (float)u;
        Y[i][1] = (float)m;
        Y[i][2] = (float)d;
        Y[i][3] = (float)r;

    }

    datafile.close();
    printf("done reading in data...\n\n");
    return Y;
}

void save_matrices(svd_data *data, float err, float eta, float reg, int max_epochs) {
    /*
     * Save the matrices so we can keep a history or continue training them
     * later. Uses a standard format based on the parameters used for training.
     */
    
    // printf("saving data...");

    // generate file names
    string file_base = "models/" + to_string(err) + "_" + to_string(K) + "_" +
                       to_string(eta) + "_" + to_string(reg) + "_" +
                       to_string(max_epochs);

    ofstream u_file(file_base + "_u.mat");
    ofstream v_file(file_base + "_v.mat");
    ofstream x_file(file_base + "_x.mat");
    ofstream x_sum_file(file_base + "_x_sum.mat");
    ofstream x_norm_file(file_base + "_x_norm.mat");
    ofstream a_file(file_base + "_a.mat");
    ofstream b_file(file_base + "_b.mat");

    // save the user matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            u_file << data->U[i][j] << " ";
        }
        u_file << "\n";
    }

    // save the movie matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            v_file << data->V[i][j] << " ";
        }
        v_file << "\n";
    }

    // save the various implicit factor matrices
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            x_file << data->X[i][j] << " ";
        }
        x_file << "\n";
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            x_sum_file << data->X_sum[i][j] << " ";
        }
        x_sum_file << "\n";
    }

    for (int i = 0; i < M; i++) {
        x_norm_file << data->X_norm[i] << " ";
    }

    // save user biases
    for (int i = 0; i < M; i++) {
        a_file << data->a[i] << " ";
    }

    // save movie biases
    for (int i = 0; i < N; i++) {
        b_file << data->b[i] << " ";
    }

    // clean up
    u_file.close();
    v_file.close();
    x_file.close();
    x_sum_file.close();
    x_norm_file.close();
    a_file.close();
    b_file.close();

    // printf("saved data...\n");
}

svd_data* load_matrices(const string file_base) {
    /*
     * Loads the user and movie matrices from the file name base and returns
     * them.
     */
    ifstream u_file(file_base + "_u.mat");
    ifstream v_file(file_base + "_v.mat");
    ifstream x_file(file_base + "_x.mat");
    ifstream x_sum_file(file_base + "_x_sum.mat");
    ifstream x_norm_file(file_base + "_x_norm.mat");
    ifstream a_file(file_base + "_a.mat");
    ifstream b_file(file_base + "_b.mat");

    float ** U = new float*[M];
    for (int i = 0; i < M; i++) {
        U[i] = new float[K];
    }

    float ** V = new float*[N];
    for (int i = 0; i < N; i++) {
        V[i] = new float[K];
    }

    float ** X = new float*[K];
    for (int i = 0; i < K; i++) {
        X[i] = new float[N];
    }

    float ** X_sum = new float*[M];
    for (int i = 0; i < M; i++) {
        X_sum[i] = new float[K];
    }

    float *X_norm = new float[M];

    float *a = new float[M];
    float *b = new float[N];

    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;
    toRet->X = X;
    toRet->X_sum = X_sum;
    toRet->X_norm = X_norm;
    toRet->a = a;
    toRet->b = b;

    // populate with saved values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            u_file >> U[i][j];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            v_file >> V[i][j];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            x_file >> X[i][j];
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            x_sum_file >> X_sum[i][j];
        }
    }

    for (int i = 0; i < M; i++) {
        x_norm_file >> X_norm[i];
    }

    for (int i = 0; i < M; i++) {
        a_file >> a[i];
    }

    for (int i = 0; i < M; i++) {
        b_file >> b[i];
    }

    u_file.close();
    v_file.close();
    x_file.close();
    x_sum_file.close();
    x_norm_file.close();
    a_file.close();
    b_file.close();

    return toRet;
}

void predict(svd_data *data, const string fbase) {
    /*
     * Given the matrices and their parameters, creates predictions and saves
     * them using a standard format.
     */
    
    printf("prediciting data...\n");
    
    // read in the test data
    float **Y;
    Y = read_data("../../data/um/qual_all.dta", NUM_QUAL_PTS);

    ofstream ofs(fbase + "_pred.txt");

    // generate and output predictions
    float pred = 0.;
    for (int ind = 0; ind < NUM_QUAL_PTS; ind++) {
        pred = prediction(data, Y[ind][0] - 1, Y[ind][1] - 1);
        if (pred < 1)
            pred = 1;
        if (pred > 5)
            pred = 5;
        ofs << pred << "\n";
    }

    // clean up
    ofs.close();

    printf("done predicting data...\n\n");

}


int main(int argc, char **argv) {

    // training parameters
    float eta = 0.007;
    float reg = 0.05;
    float eps = 0.00001;
    int max_epochs = 100;

    printf("eta: %f\nreg: %f\neps: %f\nepochs: %d\nlatent factors: %d\n\n",
            eta, reg, eps, max_epochs, K);

    
    // read in the training data
    float **Y_train;
    Y_train = read_data("../../data/um/base_all.dta", NUM_TRAIN_PTS);

    // read in the test data
    float **Y_test;
    Y_test = read_data("../../data/um/probe_all.dta", NUM_TEST_PTS);



    // and now train!
    svd_data *matrices;
    matrices = train_model(eta, reg, Y_train, Y_test, eps, max_epochs);

    // load matrices
    // matrices = load_matrices("models/0.928542_50_0.007000_0.050000_26");

    // get the out-sample error
    float err = get_err(matrices, Y_test, NUM_TEST_PTS, 0);
    printf("final error: %f\n", err);

    // save the matrices
    // save_matrices(matrices, err, eta, reg, max_epochs);

    string file_base = "predictions/" + to_string(err) + "_" + to_string(K) + "_" +
                       to_string(eta) + "_" + to_string(reg) + "_" +
                       to_string(max_epochs);

    // make predictions
    predict(matrices, file_base);


    return 0;
}
