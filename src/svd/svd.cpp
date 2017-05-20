

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <fstream>

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



svd_data* train_model(float eta, float reg, float **Y, float eps,
                      int max_epochs) {
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

    printf("done creating matricies\n\n");

    // so we can randomly go over all the points
    int *indicies = new int[NUM_PTS];
    for (int i = 0; i < NUM_PTS; i++) {
        indicies[i] = i;
    }

    ////////////////////////////////////////////////////////////////////////////

    float delta = 0.0;
    int i, j, date, Yij;
    float dot_prod;
    float before_E_in, E_in;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        before_E_in = get_err(U, V, Y, NUM_PTS, 0);

        // takes 9 seconds
        // random_shuffle(&indicies[0], &indicies[NUM_PTS-1]);

        for (int ind = 0; ind < NUM_PTS; ind++) {
            i    = Y[ind][0];
            j    = Y[ind][1];
            date = Y[ind][2];
            Yij  = Y[ind][3];

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
        E_in = get_err(U, V, Y, NUM_PTS, 0);
        printf("%f\n", E_in);

        if (epoch == 0) {
            delta = before_E_in - E_in;
        } else if (before_E_in - E_in < eps * delta) {
              break;          
        }

    }

    svd_data *toRet = (svd_data*)malloc(sizeof(svd_data));
    toRet->U = U;
    toRet->V = V;

    // clean up
    delete[] indicies;

    return toRet;

}

float **read_data(const string file, float **Y) {
    ifstream datafile(file);
    float u, m, d, r;
    for (int i = 0; i < NUM_PTS; i++) {
        datafile >> u >> m >> d >> r;

        Y[i][0] = u;
        Y[i][1] = m;
        Y[i][2] = d;
        Y[i][3] = r;
    }
    printf("done reading in data\n");
}


int main(int argc, char **argv) {

    float eta = 0.001;
    float reg = 0.01;
    float eps = 0.0001;
    int max_epochs = 20;

    float ** Y = new float*[NUM_PTS];
    for (int i = 0; i < NUM_PTS; i++) {
        Y[i] = new float[4];
    }

    read_data("../../data/um/base_all.dta", Y);


    train_model(eta, reg, Y, eps, max_epochs);

    for (int i = 0; i < NUM_PTS; i++) {
        delete Y[i];
    }
    delete[] Y;

    return 0;
}
