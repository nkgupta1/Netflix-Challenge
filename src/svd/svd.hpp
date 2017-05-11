

#define M 458293    // number of users
#define N 17770     // number of movies
#define K 30        // number of latent factors

#define NUM_PTS 102416306   // number of ratings

struct svd_data {
    float **U;
    float **V;
} ;


float dot_product(float **U, int i, float **V, int j);

float get_err(float **U, float **V, int **Y, int num_pts, float reg);

svd_data* train_model(float eta, float reg, int **Y, float eps,
                      int max_epochs);