
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <fstream>

#define M 458293    // number of users
#define N 17770     // number of movies
#define K 75        // number of latent factors

#define NUM_TRAIN_PTS  98291669   // number of ratings
#define NUM_TEST_PTS   1374739    // number of ratings
#define NUM_QUAL_PTS   2749898    // number of qual points

#define MEAN 3.608613

using namespace std;

struct svd_data {
    float **U;
    float **V;
    float *a;
    float *b;
} ;

static inline
float dot_product(float **U, int i, float **V, int j);

static inline
float prediction(svd_data *data, int i, int j);

float get_err(float **U, float **V, float *a, float *b, int **Y, int num_pts, float reg);

svd_data* train_model(float eta, float reg, float **Y, float eps,
                      int max_epochs);

float **read_data(const string file, int num_pts);

void save_matrices(svd_data *data, float err, float eta, float reg, int max_epochs);

svd_data* load_matrices(const string fbase);

void predict(svd_data *data, float err, float eta, float reg, int max_epochs);

