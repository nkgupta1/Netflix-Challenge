
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <time.h>
#include <cmath>
#include <fstream>

#define M 458293    // number of users
#define N 17770     // number of movies
#define K 30        // number of latent factors

#define NUM_PTS  94362233   // number of ratings
#define QUAL_NUM 2749898    // number of qual points

using namespace std;

struct svd_data {
    float **U;
    float **V;
} ;


float dot_product(float **U, int i, float **V, int j);

float get_err(float **U, float **V, int **Y, int num_pts, float reg);

svd_data* train_model(float eta, float reg, float **Y, float eps,
                      int max_epochs);

float **read_data(const string file, int num_pts);

void save_matrices(svd_data *data, float err, float eta, float reg, int max_epochs);

svd_data* load_matrices(const string fbase);

void predict(svd_data *data, float err, float eta, float reg, int max_epochs);


