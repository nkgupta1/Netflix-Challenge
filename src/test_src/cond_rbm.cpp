#include "cond_rbm.hpp"

// Model parameters from following paper:
// http://www.montefiore.ulg.ac.be/~glouppe/pdf/msc-thesis.pdf

#define bit(R,i,j) (*R)[((long long) i) * ((long long) M) + (long long) j]

RBM::RBM(const string file, const string v_file, 
         const string q_file, int hidden) {

    int i, j, k;

    V = 17770;
    F = hidden;
    K = 5;

    mom = 0.9;//0.5;
    // weightcost = 0.001 * 10;

    eps_w = 0.0015;//0.001 * 10;
    eps_vis = 0.0012;//0.008 * 10;
    eps_hid = 0.1;//0.002 * 10;
    eps_imp = 0.001;

    data_file = file;
    valid_file = v_file;
    qual_file = q_file;

    // Initialize default pointers (weights, biases)
    init_pointers();

    // Initialize random number generator
    init_random();

    // Initialize temporary containers used in training, predicting, validation
    init_temp();

    // Randomize weights
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            for (k = 0; k < K; k++) {
                W[i][j][k][0] = normal(gen);
                // del_W = 0 by default
                // mom_W = 0 initially
            }
        }
    }

    // Initialize visible biases (~= frequency of data)
    for (i = 0; i < V; i++) {
        vis_bias[i][0][0] = log(0.044598/(1-0.044598));
        vis_bias[i][1][0] = log(0.100438/(1-0.100438));
        vis_bias[i][2][0] = log(0.287346/(1-0.287346));
        vis_bias[i][3][0] = log(0.336991/(1-0.336991));
        vis_bias[i][4][0] = log(0.230626/(1-0.230626));
        // Frequencies of data were empirically calculated
        // for (k = 0; k < K; k++) {

            // vis_bias[i][k][0] = log(0.2/0.8);
            // vis_bias[i][k][0] = normal(gen);
            // del_vis_bias = 0 by default
            // mom_vis_bias = 0 initially
        // }
    }

    // Hidden biases initialize to zero; already handled by calloc

    // Implicit factors matrix D already 0 by calloc

    // Binary matrix R already set by init_pointers

}

RBM::RBM(const string rbm_file) {
    int i, j, k;
    ifstream ifs(rbm_file);

    // Load data file - this will allow us to reload the data
    ifs >> data_file;
    // Load valid file - this will allow us to reload the valid set
    ifs >> valid_file;
    // Load qual file - this will allow us to reload the qual set
    ifs >> qual_file;

    // Load fundamental parameters V, F, K
    ifs >> V >> F >> K;

    // Load momentum parameter
    ifs >> mom;

    // Load learning weights
    ifs >> eps_w >> eps_vis >> eps_hid >> eps_imp;;

    // Initialize default pointers (weights, biases)
    init_pointers();

    // Load weights
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            for (k = 0; k < K; k++) {
                ifs >> W[i][j][k][0];
                // Don't need to load del_W
                ifs >> W[i][j][k][2];
            }
        }
    }

    // Load visible bias
    for (i = 0; i < V; i++) {
        for (k = 0; k < K; k++) {
            ifs >> vis_bias[i][k][0];
            // Don't need to load del_vis_bias
            ifs >> vis_bias[i][k][2];
        }
    }

    // Load hidden bias
    for (j = 0; j < F; j++) {
        ifs >> hid_bias[j][0];
        // Don't need to load del_hid_bias
        ifs >> hid_bias[j][2];
    }

    // Initialize random number generator
    init_random();

    // Initialize temporary containers used in training, predicting, validation
    init_temp();

}

RBM::~RBM() {
    // Free hidden bias from inside out
    free(hid_bias[0]);
    free(hid_bias);

    // Free visible bias from inside out
    free(vis_bias[0][0]);
    free(vis_bias[0]);
    free(vis_bias);

    // Free weights
    free(W[0][0][0]);
    free(W[0][0]);
    free(W[0]);
    free(W);

    // Free implicit factors
    free(D[0][0]);
    free(D[0]);
    free(D);

    // Free binary has-rated matrix R
    // free(R[0]);
    // free(R);
    delete R;

    // Qualification set
    free(qual_idxs);
    free(qual);

    // Validation set
    free(valid_idxs);
    free(valid);

    // Training set
    free(data_idxs);
    free(data);

    // Temporary input
    free(input_t[0]);
    free(input_t);

    // Temporary hidden
    free(hidden_t);

    // Temporary cd input
    free(cd_input_t[0]);
    free(cd_input_t);

    // Temporary cd hidden
    free(cd_hidden_t);

    // Movies counter
    free(movies);
}

void RBM::save(const string file) {
    // This could be made faster with mmap - todo later
    // http://www.linuxquestions.org/questions/programming-9/
    // mmap-tutorial-c-c-511265/
    int i, j, k;
    ofstream ofs(file);

    // Save data file - this will allow us to reload the data
    ofs << data_file << " ";
    // Save valid file - this will allow us to reload the valid set
    ofs << valid_file << " ";
    // Save qual file - this will allow us to reload the qual set
    ofs << qual_file << " ";

    // Save fundamental parameters V, F, K;
    ofs << V << " " << F << " " << K << " ";

    // Save momentum parameters
    ofs << mom << " ";

    // Save learning weights
    ofs << eps_w << " " << eps_vis << " " << eps_hid << " " << eps_imp << " ";

    // Save weights
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            for (k = 0; k < K; k++) {
                ofs << W[i][j][k][0] << " ";
                // Don't need to save del_W
                ofs << W[i][j][k][2] << " ";
            }
        }
    }

    // Save visible bias
    for (i = 0; i < V; i++) {
        for (k = 0; k < K; k++) {
            ofs << vis_bias[i][k][0] << " ";
            // Don't need to save del_vis_bias
            ofs << vis_bias[i][k][2] << " ";
        }
    }

    // Save hidden bias
    for (j = 0; j < F; j++) {
        ofs << hid_bias[j][0] << " ";
        // Don't need to save del_hid_bias
        ofs << hid_bias[j][2] << " ";
    }

    // Don't need to save random number generator

    // Don't need to save temporary containers
}

/* 
 * Object contains several large matrices. These are
 * initialized here, with each in contiguous memory
 * for purposes of cache optimization
 */
void RBM::init_pointers() {
    int i, j, k;

    // Weights - Total weights, del weights, momentum weights
    W = (float ****) calloc(V, sizeof(float ***));         // Each movie
    W[0] = (float ***) calloc(V*F, sizeof(float **));      // Each factor
    W[0][0] = (float **) calloc(V*F*K, sizeof(float *));   // Each rating
    W[0][0][0] = (float *) calloc(V*F*K*3, sizeof(float)); // Full, del, mom

    // Initialize all the pointers - this format works for n-dim arrays
    for (i = 1; i < V; i++) {
        W[i] = W[i-1] + F;
        W[i][0] = W[i-1][0] + (F*K);
        W[i][0][0] = W[i-1][0][0] + (F*K*3);
    }
    for (i = 0; i < V; i++) {
        for (j = 1; j < F; j++) {
            W[i][j] = W[i][j-1] + K;
            W[i][j][0] = W[i][j-1][0] + (K*3);
        }
    }
    for (i = 0; i < V; i++) {
        for (j = 0; j < F; j++) {
            for (k = 1; k < K; k++) {
                W[i][j][k] = W[i][j][k-1] + 3;
            }
        }
    }

    // Visible Bias - Total bias, del bias, momentum weights
    vis_bias = (float ***) calloc(V, sizeof(float **));     // Each movie
    vis_bias[0] = (float **) calloc(V*K, sizeof(float *));  // Each rating
    vis_bias[0][0] = (float *) calloc(V*K*3, sizeof(float));// Full, del, mom

    // Initialize all pointers; it ain't pretty, but it works
    for (i = 1; i < V; i++) {
        vis_bias[i] = vis_bias[i-1] + K;
        vis_bias[i][0] = vis_bias[i-1][0] + (K*3);
    }
    for (i = 0; i < V; i++) {
        for (k = 1; k < K; k++) {
            vis_bias[i][k] = vis_bias[i][k-1] + 3;
        }
    }

    // Hidden bias - Total bias, del bias, momentum weights
    hid_bias = (float **) calloc(F, sizeof(float *));      // Each factor
    hid_bias[0] = (float *) calloc(F*3, sizeof(float));    // Full, del, mom

    // Initialize hidden biases. This one is a bit prettier
    for (j = 1; j < F; j++) {
        hid_bias[j] = hid_bias[j-1] + 3;
    }

    // Implicit Factors - Total bias, del bias, momentum weights
    D = (float ***) calloc(V, sizeof(float **));     // Each movie
    D[0] = (float **) calloc(V*F, sizeof(float *));  // Each rating
    D[0][0] = (float *) calloc(V*F*3, sizeof(float));// Full, del, mom

    // Initialize all pointers; it ain't pretty, but it works
    for (i = 1; i < V; i++) {
        D[i] = D[i-1] + F;
        D[i][0] = D[i-1][0] + (F*3);
    }
    for (i = 0; i < V; i++) {
        for (j = 1; j < F; j++) {
            D[i][j] = D[i][j-1] + 3;
        }
    }

    // // Binary matrix R - indexed by user, movie
    // R = (bool **) calloc(U, sizeof(bool *));      // Each factor
    // R[0] = (bool *) calloc(U*V, sizeof(bool));    // Full, del, mom

    // // Initialize binary matrix r.
    // for (i = 1; i < U; i++) {
    //     R[i] = R[i-1] + V;
    // }

    // Initialize dataset
    int N1 = line_count(data_file);
    data = (int *) calloc(4*N1, sizeof(int));

    // Id's for dataset (extra one is used for determing the end)
    data_idxs = (int *) calloc(U+1, sizeof(int));
    data_idxs[U] = N1;
    read_data(data_file, data, data_idxs, N1);

    // Initialize validation set
    int N2 = line_count(valid_file);
    valid = (int *) calloc(4*N2, sizeof(int));

    // Id's for dataset (extra one is used for determing the end)
    valid_idxs = (int *) calloc(U+1, sizeof(int));
    valid_idxs[U] = N2;
    read_data(valid_file, valid, valid_idxs, N2);

    // Initialize qualification set
    int N3 = line_count(qual_file);
    qual = (int *) calloc(4*N3, sizeof(int));

    // Id's for qual set (extra one is used for determing the end)
    qual_idxs = (int *) calloc(U+1, sizeof(int));
    qual_idxs[U] = N3;
    read_data(qual_file, qual, qual_idxs, N3);


    R = new bitset<(long long) U * (long long) M>;
    // Initialize R - simply requires read-through over all data
    init_r();
}

void RBM::init_r() {
    int i, u, m, d, r;
    // Inits binary matrix r
    ifstream dfile(data_file);
    ifstream vfile(valid_file);
    ifstream qfile(qual_file);

    int lcd = line_count(data_file);
    int lcv = line_count(valid_file);
    int lcq = line_count(qual_file);

    for (i = 0; i < lcd; i++) {
        dfile >> u >> m >> d >> r;
        bit(R,u-1,m-1) = 1;
    }

    for (i = 0; i < lcv; i++) {
        vfile >> u >> m >> d >> r;
        bit(R,u-1,m-1) = 1;
    }

    for (i = 0; i < lcq; i++) {
        qfile >> u >> m >> d >> r;
        bit(R,u-1,m-1) = 1;
    }
}

void RBM::read_data(const string file, int *dta, int *dta_ids, int lc) {
    int i, u, m, d, r, max_u;
    ifstream datafile(file);

    // Keep track of max use seen so far; use for indexing
    max_u = 0;
    for (i = 0; i < lc; i++) {
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

void RBM::init_random() {
    // Random number generator; used for weights
    random_device rd;        // Something to do with seeding
    gen = ranlux24_base(rd()); // Fast generator
    unit_rand = uniform_real_distribution<float>(0,1); // Unit uniform
    normal = normal_distribution<float>(0,0.01); // Normal u = 0, std = 1
}

void RBM::init_temp() {
    // Initialize temporary containers. These are used to hold intermediate
    // values during training, predicting, and validating. Easier
    // to init them now than to init/reinit them in every iteration.
    int i, t;

    // First, need to find out how large each temporary array must be
    // Do this by finding an upper bound for the overall maximum number of 
    // movies that are trained on/predicted/validated for any user
    int train_max = 0;
    for (i = 1; i <= U; i++) {
        t = data_idxs[i] - data_idxs[i-1];
        if (t > train_max) {
            train_max = t;
        }
    }
    int valid_max = 0;
    for (i = 1; i <= U; i++) {
        t = valid_idxs[i] - valid_idxs[i-1];
        if (t > valid_max) {
            valid_max = t;
        }
    }
    int qual_max = 0;
    for (i = 1; i <= U; i++) {
        t = qual_idxs[i] - qual_idxs[i-1];
        if (t > qual_max) {
            qual_max = t;
        }
    }
    // 2 times for in sample error
    int total_max = 2*train_max + valid_max + qual_max;

    // Initialize input_t
    input_t = (float **) calloc(total_max, sizeof(float *));
    input_t[0] = (float *) calloc(total_max*6, sizeof(float));

    // Initialize all the pointers
    for (i = 1; i < total_max; i++) {
        input_t[i] = input_t[i-1] + 6;
    }

    // Initialize hidden_t
    hidden_t = (float *) calloc(F, sizeof(float));

    // Initialize cd_input_t
    cd_input_t = (float **) calloc(train_max, sizeof(float *));
    cd_input_t[0] = (float *) calloc(train_max*6, sizeof(float));

    // Initialize the pointers
    for (i = 1; i < train_max; i++) {
        cd_input_t[i] = cd_input_t[i-1] + 6;
    }

    // Initialize cd_hidden_t
    cd_hidden_t = (float *) calloc(F, sizeof(float));

    // Initialize movies counter
    movies = (int *) calloc(V, sizeof(int));

}

void RBM::train(int start, int stop, int steps) {
    // Trains on users start to stop with steps cd steps.
    int i, j, k, l;
    int num_mov, mov;
    float eps_w_, eps_vis_, eps_hid_, eps_imp_;

    // del_W/vis_bis/hid_bias should be 0 as of previous iteration
    // And movies

    for (i = start; i <= stop; i++) {
        // Number of movies we need to consider
        num_mov = data_idxs[i] - data_idxs[i-1];

        // Initialize the array
        for (j = data_idxs[i-1]; j < data_idxs[i]; j++) {
            k = j - data_idxs[i-1];
    
            input_t[k][0] = data[4*j+1];
            input_t[k][data[4*j+3]] = 1;
        }

        // Visible -> Hidden
        forward(input_t, hidden_t, num_mov, 0, i);

        // Initialize the cd_array
        for (j = data_idxs[i-1]; j < data_idxs[i]; j++) {
            k = j - data_idxs[i-1];
    
            cd_input_t[k][0] = data[4*j+1];
            cd_input_t[k][data[4*j+3]] = 1;
        }

        // CONTRASTIVE DIVERGENCE
        forward(cd_input_t, cd_hidden_t, num_mov, 1, i); // i included for imp
        backward(cd_input_t, cd_hidden_t, num_mov, 0);

        // If steps > 1, repeat:
        for (j = 1; j < steps; j++) {
            forward(cd_input_t, cd_hidden_t, num_mov, 1, i);
            backward(cd_input_t, cd_hidden_t, num_mov, 0);
        }

        // Obtain non-discretized predictions
        forward(cd_input_t, cd_hidden_t, num_mov, 0, i);
        // backward(cd_input_t, cd_hidden_t, num_mov, 0);

        // Update del_w
        for (l = 0; l < num_mov; l++) {
            // Which movie are we updating?
            mov = (int) input_t[l][0];

            // Update movies counter
            movies[mov-1] += 1;

            // Calculate weight updates
            for (j = 0; j < F; j++) {
                for (k = 1; k <= 5; k++) {
                    W[mov-1][j][k-1][1] += hidden_t[j]*input_t[l][k] - 
                                           cd_hidden_t[j]*cd_input_t[l][k];
                }
            }
        }
        // Update del_vis_bias
        for (l = 0; l < num_mov; l++) {
            for (k = 1; k <= 5; k++) {
                mov = (int) input_t[l][0];
                vis_bias[mov-1][k-1][1] += input_t[l][k] - cd_input_t[l][k];
            }
        }
        // Update del_hid_bias
        for (j = 0; j < F; j++) {
            hid_bias[j][1] += hidden_t[j] - cd_hidden_t[j];
        }
        // Update del_implicit factors
        for (l = 0; l < num_mov; l++) {
            mov = (int) input_t[l][0];
            for (j = 0; j <= F; j++) {
                D[mov-1][j][1] += (hidden_t[j] - cd_hidden_t[j]);// * bit(R,l-1,j);
            }
        }

        // Zero out input_t, cd_input_t
        for (l = 0; l < num_mov; l++) {
            for (k = 0; k <= 5; k++) {
                input_t[l][k] = 0;
                cd_input_t[l][k] = 0;
            }
        }
        // Zero out hidden_t, cd_hidden_t
        for (j = 0; j < F; j++) {
            hidden_t[j] = 0;
            cd_hidden_t[j] = 0;
        }
    }

    // Update weights
    for (i = 0; i < V; i++) {
        // If we didn't see the movie in this subset, skip it
        if (!movies[i]) {
            continue;
        }

        eps_w_ = eps_w / movies[i];

        // If we did see the movie, update.
        for (j = 0; j < F; j++) {
            for (k = 0; k < K; k++) {
                // Update momentum
                W[i][j][k][2] = mom*W[i][j][k][2] // Current momentum
                              + eps_w_ * W[i][j][k][1]; // del_W
                              // - weightcost * W[i][j][k][0]);

                // Update actual weights
                W[i][j][k][0] += W[i][j][k][2];

                // Reset del_W
                W[i][j][k][1] = 0;
            }
        }
    }
    // Update del_implicit_factors
    for (i = 0; i < V; i++) {
        // If we didn't the the movie in this subset, we can skip it
        if (!movies[i]) {
            continue;
        }

        eps_imp_ = eps_imp / movies[i];

        // If we did, update
        for (j = 0; j < F; j++) {
            // Update momentum
            D[i][j][2] = mom*D[i][j][2] // Current momentum
                       + eps_imp_ * D[i][j][1]; // del_imp

            // Update actual implicit factors
            D[i][j][0] += D[i][j][2];

            // Reset del_implicit
            D[i][j][1] = 0;
        }
    }
    // Update visible bias
    for (i = 0; i < V; i++) {
        // If we didn't the the movie in this subset, we can skip it
        if (!movies[i]) {
            continue;
        }

        eps_vis_ = eps_vis / movies[i];

        // If we did, update
        for (k = 0; k < K; k++) {
            // Update momentum
            vis_bias[i][k][2] = mom*vis_bias[i][k][2] // Current momentum
                              + eps_vis_ * vis_bias[i][k][1]; // del_vis_bias

            // Update actual vis_bias
            vis_bias[i][k][0] += vis_bias[i][k][2];

            // Reset del_vis_bias
            vis_bias[i][k][1] = 0;
        }

        // No more need for movies[i], zero it out
        movies[i] = 0;
    }
    // Update hidden bias
    // Don't know why this should work, but I saw it somewhere
    // +1 to avoid dividing by 0
    eps_hid_ = eps_hid / ((U % (stop - start + 1)) + 1);
    for (j = 0; j < F; j++) {
        // Update momentum
        hid_bias[j][2] = mom*hid_bias[j][2] // Current momentum
                       + eps_hid_ * hid_bias[j][1]; // del_hid_bias

        // Update actual hid_bias
        hid_bias[j][0] += hid_bias[j][2];

        // Reset del_hid_bias
        hid_bias[j][1] = 0;
    }
}

void RBM::forward(float **input, float *hidden, int num_mov, bool dis, int user) {
    int i, j, k, mov;
    float sum, prob, imp_sum;
    // Calculate score for each hidden element
    for (j = 0; j < F; j++) {
        sum = 0;
        for (i = 0; i < num_mov; i++) {
            mov = (int) input[i][0];
            for (k = 1; k <= 5; k++) {
                sum += input[i][k]*W[mov-1][j][k-1][0];
            }
        }

        imp_sum = 0;
        // for (i = 0; i < V; i++) {
        //     imp_sum += bit(R,user-1,i)*D[i][j][0];
        // }

        // Scale hidden_bias by number of movies
        // Really, this makes much more sense than what I was 
        // doing before... Turn off scaling for now, add back and see
        // Maybe divide by num_mov?
        prob = sig(hid_bias[j][0] + sum + imp_sum);
        if (dis) {
            // Discretize hidden layer
            if (prob > unit_rand(gen)) {
                hidden[j] = 1.0;
            } else {
                hidden[j] = 0.0;
            }
        } else {
            hidden[j] = prob;
        }
    }
}

void RBM::backward(float **input, float *hidden, int num_mov, bool dis) {
    int i, j, k, mov;
    float sum, prob, norm;
    // Calculate score for each visible element
    for (i = 0; i < num_mov; i++) {
        mov = (int) input[i][0];
        norm = 0;
        for (k = 1; k <= 5; k++) {
            sum = 0;
            for (j = 0; j < F; j++) {
                sum += hidden[j]*W[mov-1][j][k-1][0];
            }

            // Discretize hidden layer
            prob = exp(vis_bias[mov-1][k-1][0] + sum);
            if (dis) {
                if (prob > unit_rand(gen)) {
                    input[i][k] = 1.0;
                } else {
                    input[i][k] = 0.0;
                }
            } else {
                input[i][k] = prob;
            }
            norm += prob;
        }

        for (k = 1; k <= 5; k++) {
            input[i][k] /= norm;
        }
    }
}

float RBM::validate(int start, int stop, int *dta, int *dta_ids) {
    // Trains on users start to stop with steps cd steps.
    int i, j, k, l;
    int num_known, num_pred, num_tot;

    // del_W/vis_bis/hid_bias should be 0 as of previous iteration
    // And movies

    float rmse = 0.0;
    int count = 0;
    for (i = start; i <= stop; i++) {
        // Number of movies we need to consider
        num_known = data_idxs[i] - data_idxs[i-1];
        num_pred = dta_ids[i] - dta_ids[i-1];
        num_tot = num_known + num_pred;

        // Initialize the array
        for (j = data_idxs[i-1]; j < data_idxs[i]; j++) {
            k = j - data_idxs[i-1];
    
            input_t[k][0] = data[4*j+1];
            input_t[k][data[4*j+3]] = 1;
        }
        for (j = dta_ids[i-1]; j < dta_ids[i]; j++) {
            k = (j - dta_ids[i-1]) + num_known;

            input_t[k][0] = dta[4*j+1];
        }

        // for (j = 0; j < num_mov; j++) {
        //     printf("%f,%f,%f,%f,%f,%f\n", 
        //         input_t[j][0],input_t[j][1],input_t[j][2],
        //         input_t[j][3],input_t[j][4],input_t[j][5]);
        // }

        // Run prediction (don't bother) -> Check again with 0 -> 1
        forward(input_t, hidden_t, num_tot, 0, i);
        backward(input_t, hidden_t, num_tot, 0);

        // Get validation score
        for (j = dta_ids[i-1]; j < dta_ids[i]; j++) {
            k = (j - dta_ids[i-1]) + num_known;

            // Calculate average rating and print to file.
            float numerator = 0;
            numerator += input_t[k][1] * 1;
            numerator += input_t[k][2] * 2;
            numerator += input_t[k][3] * 3;
            numerator += input_t[k][4] * 4;
            numerator += input_t[k][5] * 5;

            float denominator = 0;
            denominator += input_t[k][1];
            denominator += input_t[k][2];
            denominator += input_t[k][3];
            denominator += input_t[k][4];
            denominator += input_t[k][5];

            float mean = numerator / denominator;

            // ofs << dta[4*k]   << " "   // User
            //     << dta[4*k+1] << " "   // Movie
            //     << dta[4*k+2] << " "   // Date
            //     << mean       << "\n"; // Rating
            rmse += pow(mean - dta[4*j+3], 2);
            count++;
        }

        // Zero out input_t, cd_input_t
        for (l = 0; l < num_tot; l++) {
            for (k = 0; k <= 5; k++) {
                input_t[l][k] = 0;
            }
        }
        // Zero out hidden_t, cd_hidden_t
        for (j = 0; j < F; j++) {
            hidden_t[j] = 0;
        }
    }

    rmse = sqrt(rmse/count);
    return rmse;
}

void RBM::predict(const string outfile) {
    // Trains on users start to stop with steps cd steps.
    int i, j, k, l;
    int num_known, num_pred, num_tot;

    ofstream ofs(outfile);

    for (i = 1; i <= U; i++) {
        // Number of movies we need to consider
        num_known = data_idxs[i] - data_idxs[i-1];
        num_pred = qual_idxs[i] - qual_idxs[i-1];
        num_tot = num_known + num_pred;

        // Initialize the array
        for (j = data_idxs[i-1]; j < data_idxs[i]; j++) {
            k = j - data_idxs[i-1];
    
            input_t[k][0] = data[4*j+1];
            input_t[k][data[4*j+3]] = 1;
        }
        for (j = qual_idxs[i-1]; j < qual_idxs[i]; j++) {
            k = (j - qual_idxs[i-1]) + num_known;

            input_t[k][0] = qual[4*j+1];
        }

        // for (j = 0; j < num_mov; j++) {
        //     printf("%f,%f,%f,%f,%f,%f\n", 
        //         input_t[j][0],input_t[j][1],input_t[j][2],
        //         input_t[j][3],input_t[j][4],input_t[j][5]);
        // }

        // Run prediction (don't bother) -> Check again with 0 -> 1
        forward(input_t, hidden_t, num_tot, 0, i);
        backward(input_t, hidden_t, num_tot, 0);

        // Get validation score
        for (j = qual_idxs[i-1]; j < qual_idxs[i]; j++) {
            k = (j - qual_idxs[i-1]) + num_known;

            // Calculate average rating and print to file.
            float numerator = 0;
            numerator += input_t[k][1] * 1;
            numerator += input_t[k][2] * 2;
            numerator += input_t[k][3] * 3;
            numerator += input_t[k][4] * 4;
            numerator += input_t[k][5] * 5;

            float denominator = 0;
            denominator += input_t[k][1];
            denominator += input_t[k][2];
            denominator += input_t[k][3];
            denominator += input_t[k][4];
            denominator += input_t[k][5];

            float mean = numerator / denominator;

            ofs << mean << "\n"; // Rating
        }

        // Zero out input_t, cd_input_t
        for (l = 0; l < num_tot; l++) {
            for (k = 0; k <= 5; k++) {
                input_t[l][k] = 0;
            }
        }
        // Zero out hidden_t, cd_hidden_t
        for (j = 0; j < F; j++) {
            hidden_t[j] = 0;
        }
    }

    return;
}

int main() {
    // To run: 
    // Straight Sala
    RBM rbm = RBM("../../data/um/base_all.dta", "../../data/um/probe_all.dta", 
                  "../../data/um/qual_all.dta", 100);
    // RBM rbm = RBM("test_data/um/base_test.dta", "test_data/um/probe_test.dta", 
    //               "test_data/um/qual_test.dta", 200);
    int tsteps = 1;
    float prmse = 10;
    float rmse = 5;
    int runcount = 0;
    // rbm.mom = 0.5;

    while ((rmse < prmse) || (runcount < 15)) {
        // if (runcount == 5) {
        //     rbm.mom = 0.9;
        // }
        // if ((prmse-rmse) < 0.0001) {
        //     // Maybe also reset momentum?
        //     tsteps += 1;
        //     printf("Incrementing tsteps: now %d\n", tsteps);
        // }

        // ...
        for (int i = 0; i < 4580; i++) {
            rbm.train((i*100)+1,(i+1)*100, tsteps);
            // rbm.train(i,i,tsteps,rand_array);
            if ((i % 10) == 1) {
                printf(".");
                fflush(stdout);
            }
            if (((i+1) % 50) == 0) {
                float ein_t = rbm.validate(448001,458000, rbm.data, rbm.data_idxs);
                float rmse_t = rbm.validate(448001,458000, rbm.valid, rbm.valid_idxs);
                printf("    E_in: %f    E_valid: %f\n", ein_t, rmse_t);

            }
        }
        rbm.train(458001,458293,tsteps);

        runcount++;
        // if (runcount > 8) {
        //     rbm.eps_w *= 0.92;
        //     rbm.eps_vis *= 0.92;
        //     rbm.eps_hid *= 0.92;
        // } else if (runcount > 6) {
        //     rbm.eps_w *= 0.9;
        //     rbm.eps_vis *= 0.9;
        //     rbm.eps_hid *= 0.9;
        // } else if (runcount > 2) {
        //     rbm.eps_w *= 0.78;
        //     rbm.eps_vis *= 0.78;
        //     rbm.eps_hid *= 0.78;
        // }

        prmse = rmse;
        // rmse = rbm.validate(1,458293); Validating the whole set
        // takes 13.5 minutes (!!)
        float ein = rbm.validate(1,50000,rbm.data,rbm.data_idxs);
        rmse = rbm.validate(1,50000, rbm.valid, rbm.valid_idxs); // Should be a lot quicker
        printf("\nOverall E_in: %f    Overall RMSE: %f\n", ein, rmse);
        runcount++;

        rbm.save("cond_rbm_simple.mat");
    }
    rbm.predict("cond_rbm_simple_preds.mat");
}