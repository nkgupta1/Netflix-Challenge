
#include "svd.hpp"

using namespace std;


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

float prediction(svd_data *data, int i, int j) {
    /*
     * generates a prediction based on trained matrices
     */
    float dot_prod = dot_product(data->U, i, data->V, j);
    return dot_prod + data->a[i] + data->b[j] + MEAN;
}

float get_err(svd_data *data, float **Y, int num_pts, float reg) {
    /*
     * Get the squared error of U*V^T on Y.
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
    if (reg != 0) {
        err /= 2.0;
        err /= num_pts;

        double U_frob_norm = 0.0;
        double V_frob_norm = 0.0;
        double a_frob_norm = 0.0;
        double b_frob_norm = 0.0;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                U_frob_norm += data->U[i][j] * data->U[i][j];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                V_frob_norm += data->V[i][j] * data->V[i][j];
            }
        }

        for (int i = 0; i < M; i++) {
            a_frob_norm += data->a[i] * data->a[i];
        }

        for (int i = 0; i < N; i++) {
            b_frob_norm += data->b[i] * data->b[i];
        }

        err += 0.5 * reg * U_frob_norm;
        err += 0.5 * reg * V_frob_norm;
        err += 0.5 * reg * a_frob_norm;
        err += 0.5 * reg * b_frob_norm;
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
     * x K matrix such that Y = U * V^T + a + b
     *
     * a is user biases
     * b is movie biases
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

    float *a = new float[M];
    float *b = new float[N];

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

    for (int i = 0; i < M; i++) {
        float r = static_cast <float> (rand());
        r = r / static_cast <float> (RAND_MAX);
        r = r - 0.5;
        a[i] = r;
    }

    for (int i = 0; i < N; i++) {
        float r = static_cast <float> (rand());
        r = r / static_cast <float> (RAND_MAX);
        r = r - 0.5;
        b[i] = r;
    }

    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;
    toRet->a = a;
    toRet->b = b;

    // so we can randomly go over all the points
    int *indices = new int[NUM_TRAIN_PTS];
    for (int i = 0; i < NUM_TRAIN_PTS; i++) {
        indices[i] = i;
    }

    ////////////////////////////////////////////////////////////////////////////

    // scales eta by this every epoch
    float adaptive_learning_rate = .9;

    float delta = 0.0;
    int ind;
    int i, j, date, Yij;
    float pred;
    float before_E_in, E_in, E_out;
    float min_E_out = 100;
    
    int num_grad_issues = 0;
    float org_eta = eta;
    // this variable contains the number of times that E_out has consecutively
    // gone up
    int count_E_out_up = 0;

    printf("starting training....\n");

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // before_E_in = get_err(toRet, Y_train, NUM_TRAIN_PTS, 0);

        num_grad_issues = 0;

        // takes 9 seconds
        // random_shuffle(&indices[0], &indices[NUM_TRAIN_PTS-1]);

        for (int ind_ind = 0; ind_ind < NUM_TRAIN_PTS; ind_ind++) {
            ind = indices[ind_ind];
            i    = Y_train[ind][0];
            j    = Y_train[ind][1];
            date = Y_train[ind][2];
            Yij  = Y_train[ind][3];

            // so compiler doesn't generate warnings
            (void)date;
            (void)delta;
            (void)before_E_in;

            pred = prediction(toRet, i-1, j-1);
            if (pred < -100) {
                // printf("gradient issues!!!\n");
                num_grad_issues++;
                continue;
            }

            // update U
            for (int k = 0; k < K; k++) {
                U[i-1][k] += eta*(V[j-1][k]*(Yij - pred) - reg*U[i-1][k]);
            }
            
            // update V
            for (int k = 0; k < K; k++) {
                V[j-1][k] += eta*(U[i-1][k]*(Yij - pred) - reg*V[j-1][k]);
            }

            // update a
            a[i-1] += eta*((Yij - pred) - reg*a[i-1]);

            // update b
            b[j-1] += eta*((Yij - pred) - reg*b[j-1]);

        }

        if (num_grad_issues != 0)
            printf("num grad issues: %d\n", num_grad_issues);

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
            eta = adaptive_learning_rate * eta;
            count_E_out_up++;
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
    // delete[] indicies;

    return toRet;

}

svd_data* resume_training(svd_data* toRet, float eta, float reg, float
                      **Y_train, float **Y_test, float eps, int max_epochs) {
    /*
     * Train an SVD model using SGD.
     *
     * Factorizes a sparse M x N matrix in the product of a M x K matrix and a N
     * x K matrix such that Y = U * V^T + a + b
     *
     * a is user biases
     * b is movie biases
     *
     * eta is the learning rate.
     * reg is the regularization
     * eps is the stopping criterion
     */

    ////////////////////////////////////////////////////////////////////////////
    
    // so we can randomly go over all the points
    int *indices = new int[NUM_TRAIN_PTS];
    for (int i = 0; i < NUM_TRAIN_PTS; i++) {
        indices[i] = i;
    }

    ////////////////////////////////////////////////////////////////////////////

    // scales eta by this every epoch
    float adaptive_learning_rate = 1;

    float delta = 0.0;
    int ind;
    int i, j, date, Yij;
    float pred;
    float before_E_in, E_in, E_out;
    float min_E_out = 100;
    
    int num_grad_issues = 0;
    float org_eta = eta;
    // this variable contains the number of times that E_out has consecutively
    // gone up
    int count_E_out_up = 0;

    printf("resuming training....\n");

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // before_E_in = get_err(toRet, Y_train, NUM_TRAIN_PTS, 0);

        num_grad_issues = 0;

        // takes 9 seconds
        // random_shuffle(&indices[0], &indices[NUM_TRAIN_PTS-1]);

        for (int ind_ind = 0; ind_ind < NUM_TRAIN_PTS; ind_ind++) {
            ind = indices[ind_ind];
            i    = Y_train[ind][0];
            j    = Y_train[ind][1];
            date = Y_train[ind][2];
            Yij  = Y_train[ind][3];

            // so compiler doesn't generate warnings
            (void)date;
            (void)delta;
            (void)before_E_in;

            pred = prediction(toRet, i-1, j-1);
            if (pred < -100) {
                // printf("gradient issues!!!\n");
                num_grad_issues++;
                continue;
            }

            // update U
            for (int k = 0; k < K; k++) {
                toRet->U[i-1][k] += eta*(toRet->V[j-1][k]*(Yij - pred) - reg*toRet->U[i-1][k]);
            }
            
            // update V
            for (int k = 0; k < K; k++) {
                toRet->V[j-1][k] += eta*(toRet->U[i-1][k]*(Yij - pred) - reg*toRet->V[j-1][k]);
            }

            // update a
            toRet->a[i-1] += eta*((Yij - pred) - reg*toRet->a[i-1]);

            // update b
            toRet->b[j-1] += eta*((Yij - pred) - reg*toRet->b[j-1]);

        }

        if (num_grad_issues != 0)
            printf("num grad issues: %d\n", num_grad_issues);

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
            eta = adaptive_learning_rate * eta;
            count_E_out_up++;
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

    float *a = new float[M];
    float *b = new float[N];

    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;
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

    for (int i = 0; i < M; i++) {
        a_file >> a[i];
    }

    for (int i = 0; i < M; i++) {
        b_file >> b[i];
    }

    u_file.close();
    v_file.close();
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


void predict_probe(svd_data *data, const string fbase) {
    /*
     * Given the matrices and their parameters, creates predictions and saves
     * them using a standard format.
     */
    
    printf("prediciting data...\n");
    
    // read in the test data
    float **Y;
    Y = read_data("../../data/um/probe_all.dta", NUM_TEST_PTS);

    ofstream ofs(fbase + "_probe.txt");

    // generate and output predictions
    float pred = 0.;
    for (int ind = 0; ind < NUM_TEST_PTS; ind++) {
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
    int max_epochs = 4;

    printf("eta: %f\nreg: %f\neps: %f\nepochs: %d\nlatent factors: %d\n\n",
            eta, reg, eps, max_epochs, K);

    
    // read in the training data
    float **Y_train;
    Y_train = read_data("../../data/um/base+probe_all.dta", NUM_TRAIN_PTS);

    // read in the test data
    float **Y_test;
    Y_test = read_data("../../data/um/probe_all.dta", NUM_TEST_PTS);



    // and now train!
    svd_data *matrices;
    // matrices = train_model(eta, reg, Y_train, Y_test, eps, max_epochs);

    // load matrices
    matrices = load_matrices("models/0.917174_50_0.007000_0.050000_71");

    matrices = resume_training(matrices, eta, reg, Y_train, Y_test, eps, max_epochs);


    // get the out-sample error
    float err = get_err(matrices, Y_test, NUM_TEST_PTS, 0);
    printf("final error: %f\n", err);

    // save the matrices
    // save_matrices(matrices, err, eta, reg, max_epochs);

    string file_base = "predictions/base+probe_" + to_string(err) + "_" + to_string(K) + "_" +
                       to_string(eta) + "_" + to_string(reg) + "_" +
                       to_string(max_epochs);

    // make predictions
    predict(matrices, file_base);
    // predict_probe(matrices, file_base);


    return 0;
}