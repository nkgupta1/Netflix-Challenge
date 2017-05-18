/*!
 * Implementation of restrcited boltzmann machine.
 */

#include "rbm.hpp"

#define U   458293      // Users
// #define U   23          // Users
#define M   17770       // Movies
// #define N   102416306   // Data Points
// #define N   4179        // Data Points in function testing set
#define D   2243        // Number days
// #define K  5           // Number ratings

using namespace std;

// Normal constructor
RBM::RBM(const string file, int hidden, float learning_rate) {
    int i, j, k;
    // Parameters of RBM
    V = M;
    F = hidden;
    K = 5;
    eps = learning_rate;
    data_file = file;
    
    // Handle default initialization in its own function;
    // makes life easier for saving / loading
    init(data_file);

    // Weights
    W = (float ***) malloc(sizeof(float **) * V);
    for (i = 0; i < V; i++) {
        W[i] = (float **) malloc(sizeof(float *) * F);
        for (j = 0; j < F; j++) {
            W[i][j] = (float *) malloc(sizeof(float) * K);
            for (k = 0; k < K; k++) {
                W[i][j][k] = sym_rand(gen);
            }
        }
    }

    // Visible biases
    vis_bias = (float **) malloc(sizeof(float *) * V);
    for (i = 0; i < V; i++) {
        vis_bias[i] = (float *) malloc(sizeof(float) * K);
        for (j = 0; j < K; j++) {
            vis_bias[i][j] = sym_rand(gen);
        }
    }

    // Hidden biases
    hid_bias = (float *) malloc(sizeof(float) * F);
    for (i = 0; i < F; i++) {
        hid_bias[i] = sym_rand(gen);
    }
}

// Load constructor; loads RBM from file, all is good.
RBM::RBM(const string file) {
    int i, j, k;
    ifstream ifs(file);

    // Load in parameters
    ifs >> V >> F >> K >> eps >> data_file;

    // Initialize 
    init(data_file);

    // Weights
    W = (float ***) malloc(sizeof(float **) * V);
    for (i = 0; i < V; i++) {
        W[i] = (float **) malloc(sizeof(float *) * F);
        for (j = 0; j < F; j++) {
            W[i][j] = (float *) malloc(sizeof(float) * K);
            for (k = 0; k < K; k++) {
                ifs >> W[i][j][k];
            }
        }
    }

    // Visible biases
    vis_bias = (float **) malloc(sizeof(float *) * V);
    for (i = 0; i < V; i++) {
        vis_bias[i] = (float *) malloc(sizeof(float) * K);
        for (j = 0; j < K; j++) {
            ifs >> vis_bias[i][j];
        }
    }

    // Hidden biases
    hid_bias = (float *) malloc(sizeof(float) * F);
    for (i = 0; i < F; i++) {
        ifs >> hid_bias[i];
    }
}

void RBM::save(const string fname) {
    int i, j, k;
    ofstream ofs(fname);

    // Save parameters
    ofs << V << " " << F << " " << K << " " << eps << " " << data_file << " ";

    // Weights
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            for (k = 0; k < K; k++) {
                ofs << W[i][j][k] << " ";
            }
        }
    }

    // Visible biases
    for (i = 0; i < V; i++) {
        for (j = 0; j < K; j++) {
            ofs << vis_bias[i][j] << " ";
        }
    }

    // Hidden biases
    for (i = 0; i < F; i++) {
        ofs << hid_bias[i] << " ";
    }    
}

RBM::~RBM(){
    // Handle in another function. Makes life easier for saving, loading
    // Delete data array, indices
    deinit();
}

// Initialize data set. Called at construction and at load.
void RBM::init(const string file) {
    printf("   Initializing arrays...\n");
    // Dataset
    int N = line_count(file);
    data = (int *) malloc(sizeof(int) * 4 * N);
    memset(data, 0, sizeof(int)*4*N);

    // Id's for dataset (extra one is used for determing the end)
    data_idxs = (int *) malloc(sizeof(int) * (U+1));
    memset(data_idxs, 0, sizeof(int)*U);
    data_idxs[U] = N;
    read_data(file, data, data_idxs);
    printf("   Arrays initialized.\n");

    // Random number generator; used for weights
    random_device rd;        // Something to do with seeding
    gen = ranlux24_base(rd()); // Fast generator
    sym_rand = uniform_real_distribution<float>(-0.01,0.01); // Uniform distribution
    unit_rand = uniform_real_distribution<float>(0,1); // Unit uniform
}

void RBM::deinit() {
    free(data);
    free(data_idxs);

    // Delete weights, inward out
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < F; j++) {
            free(W[i][j]);
        }
        free(W[i]);
    }
    free(W);

    // Delete visible biases
    for (int i = 0; i < V; i++) {
        free(vis_bias[i]);
    }
    free(vis_bias);

    // Delete hidden biases
    free(hid_bias);
}

void RBM::read_data(const string file, int *dta, int *dta_ids) {
    int i, u, m, d, r, max_u, N;
    ifstream datafile(file);
    N = line_count(file);

    // Keep track of max use seen so far; use for indexing
    max_u = 0;
    for (i = 0; i < N; i++) {
        datafile >> u >> m >> d >> r;

        dta[4*i+0] = u;
        dta[4*i+1] = m;
        dta[4*i+2] = d;
        dta[4*i+3] = r;

        if (u > max_u) {
            dta_ids[u-1] = i;
            max_u = u;
        }
    }

    // Dta_ids encounters issues if users are skipped.
    // Fix those here.
    for (i = 1; i < U; i++) {
        if (dta_ids[i] == 0) {
            dta_ids[i] = dta_ids[i-1];
        }
    }
}

// Trains users from start to stop with steps contrastive
// divergence steps.
void RBM::train(int start, int stop, int steps) {
    int i, j, k, l, mov;

    // CHANGE IN WEIGHTS, BIASES

    // Weights
    float ***del_W = (float ***) malloc(sizeof(float **) * V);
    for (i = 0; i < V; i++) {
        del_W[i] = (float **) malloc(sizeof(float *) * F);
        for (j = 0; j < F; j++) {
            del_W[i][j] = (float *) malloc(sizeof(float) * K);
            for (k = 0; k < K; k++) {
                del_W[i][j][k] = 0;
            }
        }
    }
    // printf("Weights good.\n");
    // Visible biases
    float **del_vis_bias = (float **) malloc(sizeof(float *) * V);
    for (i = 0; i < V; i++) {
        del_vis_bias[i] = (float *) malloc(sizeof(float) * K);
        for (j = 0; j < K; j++) {
            del_vis_bias[i][j] = 0;
        }
    }
    // printf("Vis biases good.\n");
    // Hidden biases
    float *del_hid_bias = (float *) malloc(sizeof(float) * F);
    for (i = 0; i < F; i++) {
        del_hid_bias[i] = 0;
    }
    // printf("Hidden biases good.\n");

    // MINIBATCH LOOP
    for (i = start; i <= stop; i++) {
        // printf("Starting user %d\n", start);
        // This term is used a lot; num_mov per user
        int num_mov = data_idxs[i]-data_idxs[i-1];

        // INITIALIZE ARRAYS

        // Row for each rated movie, one column for mov_id, 5 for ratings
        float **input = (float **) malloc(sizeof(float *) * num_mov);
        for (j = 0; j < num_mov; j++) {
            input[j] = (float *) malloc(sizeof(float) * 6);
            memset(input[j], 0, sizeof(float) * 6);
        }
        // Hidden results for h_data
        float *hidden = (float *) malloc(sizeof(float) * F);
        memset(hidden, 0, sizeof(float) * F);
        // printf("Allocation good.\n");

        // Populate input, calculate hidden state
        user_to_input(i,input);
        forward(input, hidden, num_mov, 0);
        // printf("U-to-i good.\n");

        // Now, create arrays for contrastive divergence. The input will look the
        // same as the initial input, but will be modified
        float **cd_input = (float **) malloc(sizeof(float *) * num_mov);
        for (j = 0; j < num_mov; j++) {
            cd_input[j] = (float *) malloc(sizeof(float) * 6);
            memset(cd_input[j], 0, sizeof(float) * 6);
        }
        user_to_input(i,cd_input);
        // Ditto for the hidden values
        float *cd_hidden = (float *) malloc(sizeof(float) * F);
        memset(cd_hidden, 0, sizeof(float) * F);

        // printf("Copy objects good.\n");

        // CONTRASTIVE DIVERGENCE

        // We do it at least once...
        forward(cd_input, cd_hidden, num_mov, 1);
        backward(cd_input, cd_hidden, num_mov, 0);
        // ... and then potentially more
        for (j = 1; j < steps; j++) {
            forward(cd_input, cd_hidden, num_mov, 1);
            backward(cd_input, cd_hidden, num_mov, 0);
        }

        // For last sampling, don't discretize
        forward(cd_input, cd_hidden, num_mov, 0);

        // printf("CD good.\n");

        // ADD TO UPDATES

        // Weights
        for (j = 0; j < num_mov; j++) {
            mov = (int) input[j][0];
            for (l = 0; l < F; l++) {
                for (k = 1; k <= 5; k++) {
                    del_W[mov-1][l][k-1] += eps*(hidden[l]*input[j][k] - cd_hidden[l]*cd_input[j][k]);
                }
            }
        }
        // Visible Biases
        for (j = 0; j < num_mov; j++) {
            for (k = 1; k <= 5; k++) {
                mov = (int) input[j][0];
                del_vis_bias[mov-1][k-1] += eps*(input[j][k]-cd_input[j][k]);
            }
        }
        // Hidden Biases
        for (j = 0; j < F; j++) {
            del_hid_bias[j] += eps*(hidden[j]-cd_hidden[j]);
        }
        // printf("Adding good.\n");

        // for (int j = 0; j < num_mov; j++) {
        //     printf("%f,%f,%f,%f,%f,%f\n", 
        //         cd_input[j][0],cd_input[j][1],cd_input[j][2],cd_input[j][3],
        //         cd_input[j][4],cd_input[j][5]);
        // }
        // printf("\n\n");

        // CLEAN-UP

        for (j = 0; j < num_mov; j++) {
            free(input[j]);
        }
        free(input);
        free(hidden);

        for (j = 0; j < num_mov; j++) {
            free(cd_input[j]);
        }
        free(cd_input);
        free(cd_hidden);

        // printf("CLEAN-UP good.\n");
    }

    // UPDATE PARAMETERS

    // Weights
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < F; j++) {
            for (int k = 0; k < K; k++) {
                W[i][j][k] += del_W[i][j][k];
            }
        }
    }
    // Visible biases
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < K; j++) {
            vis_bias[i][j] += del_vis_bias[i][j];
        }
    }
    // Hidden biases
    for (int i = 0; i < F; i++) {
        hid_bias[i] += (del_hid_bias[i]/(stop-start+1));
    }

    // printf("Overall update good.\n");

    // CLEAN-UP

    // Weights
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            free(del_W[i][j]);
        }
        free(del_W[i]);
    }
    free(del_W);
    // Visible biases
    for (i = 0; i < V; i++) {
        free(del_vis_bias[i]);
    }
    free(del_vis_bias);

    // Hidden biases
    free(del_hid_bias);

    // printf("Then why'd you segfault?\n");
}

// Given a user, transitions that user into 
// ONLY WORKS FOR KNOWN SCORES
void RBM::user_to_input(int u, float **input) {
    for (int i = data_idxs[u-1]; i < data_idxs[u]; i++) {
        int j = i - data_idxs[u-1];

        input[j][0] = data[4*i+1];
        input[j][data[4*i+3]] = 1;
    }
}

// Input -> Hidden
void RBM::forward(float **input, float *hidden, int num_mov, bool dis) {
    int i, j, k, mov;
    float sum, prob;
    // Calculate score for each hidden element
    for (i = 0; i < F; i++) {
        sum = 0;
        for (j = 0; j < num_mov; j++) {
            mov = (int) input[j][0];
            for (k = 1; k <= 5; k++) {
                sum += input[j][k]*W[mov-1][i][k-1];
            }
        }

        prob = sig(hid_bias[i] + sum);
        if (dis) {
            // Discretize hidden layer
            if (prob > unit_rand(gen)) {
                hidden[i] = 1.0;
            } else {
                hidden[i] = 0.0;
            }
        } else {
            hidden[i] = prob;
        }
    }
}

// Input <- Hidden
void RBM::backward(float **input, float *hidden, int num_mov, bool dis) {
    int i, j, k, mov;
    float sum, prob;
    // Calculate score for each visible element
    for (j = 0; j < num_mov; j++) {
        mov = (int) input[j][0];
        for (k = 1; k <= 5; k++) {
            sum = 0;
            for (i = 0; i < F; i++) {
                sum += hidden[i]*W[mov-1][i][k-1];
            }

            // Discretize hidden layer
            prob = sig(vis_bias[mov-1][k-1] + sum);
            if (dis) {
                if (prob > unit_rand(gen)) {
                    input[j][k] = 1.0;
                } else {
                    input[j][k] = 0.0;
                }
            } else {
                input[j][k] = prob;
            }
        }
    }
}

// Fname = qual, outname = where it gets printed
void RBM::predict(const string fname, const string outname, int start, int stop) {
    int i, j, k, num_known, num_pred, num_tot;
    float **input, *hidden;
    ofstream ofs(outname);

    // printf("Does this fail?\n");

    // Dataset
    int N = line_count(fname);
    int *dta = (int *) malloc(sizeof(int) * 4 * N);
    memset(dta, 0, sizeof(int)*4*N);

    // Id's for dataset (extra one is used for determing the end)
    int *dta_ids = (int *) malloc(sizeof(int) * (U+1));
    memset(dta_ids, 0, sizeof(int)*U);
    dta_ids[U] = N;
    read_data(fname, dta, dta_ids);

    // For each user, generate predictions
    for (i = start; i <= stop; i++) {
        // Alright, now have data. Time to predict!
        num_known = data_idxs[i]-data_idxs[i-1];
        num_pred = dta_ids[i] - dta_ids[i-1];
        num_tot = num_known + num_pred;

        // printf("First population good.\n");
        input = (float **) malloc(sizeof(float *) * num_tot);
        while (input == NULL) {
            exit(0);
        }
        for (j = 0; j < num_tot; j++) {
            input[j] = (float *) malloc(sizeof(float) * 6);
            memset(input[j], 0, sizeof(float) * 6);
        }
        // Hidden results for h_data
        hidden = (float *) malloc(sizeof(float) * F);
        memset(hidden, 0, sizeof(float) * F);

        // Populate input with known movies...
        for (k = data_idxs[i-1]; k < data_idxs[i]; k++) {
            j = k - data_idxs[i-1];

            input[j][0] = data[4*k+1];
            input[j][data[4*k+3]] = 1;
        }
        // ... and with unknown movies
        for (k = dta_ids[i-1]; k < dta_ids[i]; k++) {
            j = (k - dta_ids[i-1]) + num_known;

            input[j][0] = dta[4*k+1];
        }

        // printf("Second population good.\n");

        // Run predictions
        forward(input, hidden, num_tot, 1);
        backward(input, hidden, num_tot, 0);
        // for (j = 0; j < num_tot; j++) {
        //     printf("%f,%f,%f,%f,%f,%f\n", 
        //         input[j][0],input[j][1],input[j][2],input[j][3],
        //         input[j][4],input[j][5]);
        // }
        // exit(0);

        // printf("Forward/backward good\n");

        for (k = dta_ids[i-1]; k < dta_ids[i]; k++) {
            j = (k - dta_ids[i-1]) + num_known;

            // Calculate average rating and print to file.
            float numerator = 0;
            numerator += input[j][1] * 1;
            numerator += input[j][2] * 2;
            numerator += input[j][3] * 3;
            numerator += input[j][4] * 4;
            numerator += input[j][5] * 5;

            float denominator = 0;
            denominator += input[j][1];
            denominator += input[j][2];
            denominator += input[j][3];
            denominator += input[j][4];
            denominator += input[j][5];

            float mean = numerator / denominator;

            // ofs << dta[4*k]   << " "   // User
            //     << dta[4*k+1] << " "   // Movie
            //     << dta[4*k+2] << " "   // Date
            //     << mean       << "\n"; // Rating
            ofs << mean << "\n"; // Don't really need extra info
        }

        // Clean-up
        for (j = 0; j < num_tot; j++) {
            free(input[j]);
        }
        free(input);
        free(hidden);

    }

    free(dta);
    free(dta_ids);
}

int RBM::line_count(const string fname) {
    // Counts number of lines in a file. Going into util at some point
    int count = 0;
    ifstream ifs(fname);
    string temp;

    while(getline(ifs, temp)) {
        count++;
    }
    return count;
}

float RBM::sig(float num) {
    return (1.0 / (1.0 + exp(-num)));
}

// Trains and evaluates the whole thing. Uses Python-like argument
// syntax because why not.
// Written as a constructor because I don't want to write it as 
// a static method
RBM::RBM(const string file, int hidden, float learning_rate,
    const string fname, const string outname, const string save_name,
    int full_iters, int cd_steps) {

    int i, j;

    // Create the RBM
    printf("Creating RBM...\n");
    RBM rbm = RBM(file, hidden, learning_rate);
    printf("RBM Created.\n");

    for (i = 0; i < full_iters; i++) {
        printf("Beginning iteration %d of %d\n", i+1, full_iters);

        // 916 = 458000 / 500
        for (j = 0; j < 4582; j++) {
            rbm.train((j*100)+1,(j+1)*100,cd_steps);
            // Every 1000, print a . just to prove everything's running
            if (((j+1)*100) % 1000 == 0) {
                printf(".");
                fflush(stdout);
            }
            // Every 50000, give an update
            if ((((j+1)*100) % 50000 == 0) && (((j+1)*100) != 450000)) {
                printf("    %d/458293 completed.\n", (j+1)*100);
            }
            // Every 150000, save progress
            if ((((j+1)*100) % 150000 == 0) && (((j+1)*100) != 450000)) {
                printf("    Saving...\n");
                rbm.save(save_name);
                // rbm.predict(fname,outname,1,U);
                printf("    Saved.\n");
            }
        }
        // Train on last few examples
        rbm.train(458201,458293,cd_steps);
        printf("    458293/458293 completed.\n");
        printf("    Printing sand saving...\n");
        rbm.save(save_name);
        rbm.predict(fname,outname,1,U);
        printf("    Printed and saved.\n");

    }
    printf("All done!\n");
}

int main() {
    // To-Do: 
    // Validation

    // RBM f = RBM("../../data/um/base_all.dta", 100, 0.01);
    // f.predict("../../data/um/qual_all.dta", "../../data/um/rbm_preds.dta",1,U);
    RBM("../../data/um/base_all.dta", 200, 0.01, "../../data/um/qual_all.dta", 
        "../../data/um/rbm_preds.dta", "rbm.mat", 10, 3);
    // RBM rbm = RBM("rbm.mat");
    // printf("Loaded.\n");
    // rbm.predict("../../data/um/qual_all.dta", "../../data/um/rbm_preds.dta");
    // printf("Finished.\n");
    // RBM x = RBM("test_data/um/base_test.dta", 100, 0.1);
    // for (int i = 0; i < 1; i++) {
    //     x.train(3,3,3);
    // }
    // x.predict("test_data/um/base_test.dta", "test_data/um/qual_preds.dta", 3, 3);
}