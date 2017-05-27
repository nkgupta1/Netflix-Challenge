#ifndef __RBM_H__
#define __RBM_H__

// One of these has a 72 kb memory leak...
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

// Random also appears to have memory leaks. Tsk tsk.
#include <random>

#include <cmath>
#include <unistd.h>
#include <assert.h>
#include <algorithm>

#include "util.hpp"

#define U   458293      // Users
#define M   17770       // Movies
#define D   2243        // Number days

using namespace std;

class RBM {
public:
    // File containing data
    string data_file;
    // Training data
    int *data;
    // Training data indices
    int *data_idxs;

    // File containing validation set
    string valid_file;
    // Validation data
    int *valid;
    // Validation data indices
    int *valid_idxs;

    // File containing qual data points
    string qual_file;
    // Qual data;
    int *qual;
    // Qual data indices
    int *qual_idxs;

    // Number of movies
    int V;
    // Number of hidden features
    int F;
    // Number of possible ratings
    int K;

    // Current momentum
    float mom;
    // Weightcost - penalize large weights -> encourage sparsity
    float weightcost;
    // TODO - Sparsity target, consider more sophisticated weightcost
    // TODO - see if momentum should be applied even without signal

    // Random uniform number generator
    ranlux24_base gen;
    uniform_real_distribution<float> unit_rand;
    normal_distribution<float> normal;

    // Weights, del_weights, momentum for weights
    float ****W;
    // Vis bias, del_vis_bias, momentum for visible biases
    float ***vis_bias;
    // Hidden bias, del_hid_bias, momentum for hidden bias
    float **hid_bias;

    // Temporary input container
    float **input_t;
    // Temporary hidden container
    float *hidden_t;
    // Temporary contrastive divergence input container
    float **cd_input_t;
    // Temporary contrastive divergence hidden container
    float *cd_hidden_t;
    // Temporary container; keeps track if we need to update a specific movie
    int *movies;

    // Learning rate for weights
    float eps_w;
    // Learning rate for visible biases
    float eps_vis;
    // Learning rate for hidden biases (this is a per-unit factor)
    float eps_hid;

    // CONSTRUCTORS

    // Default constructor
    RBM(const string file, const string v_file, 
        const string q_file, int hidden);
    // Copy constructor
    RBM(const string rbm_file);
    ~RBM();

    // SAVE - Load with copy constructor
    void save(const string fname);

    // INITIALIZING FUNCTIONS

    // Pointers
    void init_pointers();
    void read_data(const string file, int *dta, int *dta_ids, int lc);

    // RNG
    void init_random();

    // Temporary files
    void init_temp();

    // TRAIN

    // Overall function
    void train(int start, int stop, int steps, int *rand_array);
    // Input -> Hidden
    void forward(float **input, float *hidden, int num_mov, bool dis);
    // Hidden -> Input
    void backward(float **input, float *hidden, int num_mov, bool dis);

    // VALIDATE
    float validate(int start, int stop, int *dta, int *dta_ids);

};

#endif