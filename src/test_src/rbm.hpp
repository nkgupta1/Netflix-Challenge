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

#define U   458293      // Users
// #define U   23          // Users
#define M   17770       // Movies
// #define N   102416306   // Data Points
// #define N   4179        // Data Points in function testing set
#define D   2243        // Number days
// #define K  5           // Number ratings

using namespace std;

class RBM {
public:
    // Complete dataset
    int *data;
    // Indices for where each person's info starts on the dataset
    int *data_idxs;

    // Number of movies
    int V;
    // Number of hidden units
    int F;
    // Number of possible ratings
    int K;

    // Weights
    float ***W;
    // Visible biases
    float **vis_bias;
    // Hidden biase
    float *hid_bias;

    // Learning rate
    float eps;

    // Data file
    string data_file;

    // Random uniform number generator
    ranlux24_base gen;
    uniform_real_distribution<float> sym_rand;
    uniform_real_distribution<float> unit_rand;

    RBM(const string file, int hidden, float learning_rate);
    RBM(const string file);
    ~RBM();

    // Run constructor. The one you should actually use
    RBM(const string file, int hidden, float learning_rate,
    const string fname, const string outname, const string save_name,
    int full_iters, int cd_steps);

    void init(const string file);
    void deinit();
    void save(const string fname);

    // Reads in initial data - probably put this in util at some point
    void read_data(const string file, int *dta, int *dta_ids);

    void train(int start, int stop, int steps);
    void user_to_input(int u, float **input);

    void predict(const string fname, const string outname, int start, int stop);

    float sig(float num);
    int line_count(const string fname);

    // For a given user, calculates hidden from visible
    void forward(float **input, float *hidden, int num_mov, bool dis);
    void backward(float **input, float *hidden, int num_mov, bool dis);
};

#endif