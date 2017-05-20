
#include "svd.hpp"

float dot_product(float **U, int i, float **V, int j) {
    /*
     * Takes the dot product of 2 rows of U and V.
     */
    
    float dot_prod = 0.;
    for (int k = 0; k < K; k++) {
        dot_prod += U[i][k] * V[j][k];
    }
    return dot_prod;
}

float get_err(float **U, float **V, float **Y, int num_pts, float reg) {
    /*
     * Get the squared error of U*V^T on Y.
     */
    double err = 0.0;
    float dot_prod;
    float i, j, Yij;

    // calculate the mean squared error on each training point
    for (int ind = 0; ind < num_pts; ind++) {
        i    = Y[ind][0];
        j    = Y[ind][1];
        Yij  = Y[ind][3];

        dot_prod = dot_product(U, i-1, V, j-1);

        err += (Yij - dot_prod) * (Yij - dot_prod);
    }

    // if reg is not 0, then we are using this error for training so include
    // regularization
    if (reg != 0) {
        err /= 2.0;
        err /= num_pts;

        double U_frob_norm = 0.0;
        double V_frob_norm = 0.0;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                U_frob_norm += U[i][j] * U[i][j];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                V_frob_norm += V[i][j] * V[i][j];
            }
        }

        err += 0.5 * reg * U_frob_norm;
        err += 0.5 * reg * V_frob_norm;
    } else {
        // if reg is 0, we are using this error for testing so want RMSE error
        err /= num_pts;
        err = sqrt(err);
    }

    return err;

}



svd_data* train_model(float eta, float reg, float **Y_train, float **Y_test,
                      float eps, int max_epochs) {    
    /*
     * Train an SVD model using SGD.
     *
     * Factorizes a sparse M x N matrix in the product of a M x K matrix and a N
     * x K matrix such that Y = U * V^T
     *
     * eta is the learning rate.
     * reg is the regularization
     * eps is the stopping criterion
     */

    ////////////////////////////////////////////////////////////////////////////
    
    srand(time(NULL));



    float ** U = new float*[M];
    for (int i = 0; i < M; i++) {
        U[i] = new float[K];
    }

    float ** V = new float*[N];
    for (int i = 0; i < N; i++) {
        V[i] = new float[K];
    }

    // populate with random values
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float r = static_cast <float> (rand());
            r = r / static_cast <float> (RAND_MAX);
            r = r - 0.5;
            U[i][j] = r;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float r = static_cast <float> (rand()); 
            r = r / static_cast <float> (RAND_MAX);
            r = r - 0.5;
            V[i][j] = r;
        }
    }

    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;

    printf("done creating matrices...\n\n");

    // so we can randomly go over all the points
    // int *indicies = new int[NUM_PTS];
    // for (int i = 0; i < NUM_PTS; i++) {
    //     indicies[i] = i;
    // }

    ////////////////////////////////////////////////////////////////////////////

    float delta = 0.0;
    int i, j, date, Yij;
    float dot_prod;
    float before_E_in, E_in, E_out;

    printf("starting training....\n");

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        before_E_in = get_err(U, V, Y_train, NUM_TRAIN_PTS, 0);

        // takes 9 seconds
        // random_shuffle(&indicies[0], &indicies[NUM_TRAIN_PTS-1]);

        for (int ind = 0; ind < NUM_TRAIN_PTS; ind++) {
            i    = Y_train[ind][0];
            j    = Y_train[ind][1];
            date = Y_train[ind][2];
            Yij  = Y_train[ind][3];

            // so compiler doesn't generate warnings
            (void)date;

            dot_prod = dot_product(U, i-1, V, j-1);

            // update U
            for (int k = 0; k < K; k++) {
                U[i-1][k] += eta*V[j-1][k]*(Yij - dot_prod) - reg*eta*U[i-1][k];
            }
            
            // update V
            for (int k = 0; k < K; k++) {
                V[j-1][k] += eta*U[i-1][k]*(Yij - dot_prod) - reg*eta*V[j-1][k];
            }

        }

        // get the error for the epoch
        E_in  = get_err(U, V, Y_train, NUM_TRAIN_PTS, 0);
        E_out = get_err(U, V, Y_train, NUM_TRAIN_PTS, 0);
        printf("iteration %3d:\tE_in: %1.5f\tE_out: %1.5f\n", epoch, E_in, E_out);

        if (epoch % 10 == 0) {
            save_matrices(toRet, E_in, eta, reg, epoch);
        }

        // check termination condition
        if (epoch == 0) {
            delta = before_E_in - E_in;
        } else if (before_E_in - E_in < eps * delta) {
              break;          
        }

    }

    printf("finished training...\n\n");


    // clean up
    // delete[] indicies;

    return toRet;

}

float **read_data(const string file, int num_pts) {
    /*
     * Read in data for training or test.
     * In format:
     *     user, movie, date, rating
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
    float u, m, d, r;
    for (int i = 0; i < num_pts; i++) {
        datafile >> u >> m >> d >> r;

        Y[i][0] = u;
        Y[i][1] = m;
        Y[i][2] = d;
        Y[i][3] = r;
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

    // clean up
    u_file.close();
    v_file.close();
    printf("saved data...\n");
}

svd_data* load_matrices(const string fbase) {
    /*
     * Loads the user and movie matrices from the file name base and returns
     * them.
     */
    return NULL;
}

void predict(svd_data *data, float err, float eta, float reg, int max_epochs) {
    /*
     * Given the matrices and their parameters, creates predictions and saves
     * them using a standard format.
     */
    
    printf("prediciting data...\n");
    
    // read in the test data
    float **Y;
    Y = read_data("../../data/um/qual_all.dta", NUM_QUAL_PTS);

    // create filename
    string pred_name = "predictions/" + to_string(err) + "_" + to_string(K) +
                       "_" + to_string(eta) + "_" + to_string(reg) + "_" +
                       to_string(max_epochs) + "_pred.txt";

    ofstream ofs(pred_name);

    // generate and output predictions
    float pred = 0.;
    for (int ind = 0; ind < NUM_QUAL_PTS; ind++) {
        pred = dot_product(data->U, Y[ind][0] - 1, data->V, Y[ind][1] - 1);
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
    float eta = 0.001;
    float reg = 0.01;
    float eps = 0.00001;
    int max_epochs = 100;

    
    // read in the training data
    float **Y_train;
    Y_train = read_data("../../data/um/base_all.dta", NUM_TRAIN_PTS);

    // read in the test data
    float **Y_test;
    Y_test = read_data("../../data/um/valid_all.dta", NUM_TRAIN_PTS);



    // and now train!
    svd_data *matrices;
    matrices = train_model(eta, reg, Y_train, Y_test, eps, max_epochs);

    // get the in-sample error
    float err = get_err(matrices->U, matrices->V, Y_train, NUM_TRAIN_PTS, 0);

    // save the matrices
    save_matrices(matrices, err, eta, reg, max_epochs);

    // make predictions
    predict(matrices, err, eta, reg, max_epochs);


    // CLEAN UP
    for (int i = 0; i < NUM_TRAIN_PTS; i++) {
        delete Y_train[i];
    }
    delete[] Y_train;

    return 0;
}
