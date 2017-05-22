// #include "util.hpp"
//#include "preprocess.hpp"
#include "knn.hpp"
#include <time.h>

int main () {
    clock_t begin = clock();

    //1. Read in data
    printf("Hashing\n");
    vector<double> *user_info;
    user_info = (vector<double> *)malloc((U+1) * sizeof(vector<double>));
    hash_info("../../../um/processed_base_all.dta", user_info);
    //hash_info("../../../um/base_processed.dta", user_info);

    //2. Get Q users with highest number of ratings
    printf("Getting top q\n");
    int top_q[Q];
    get_top_q(user_info, top_q);

    //3. Compute ratings for infile points
    ifstream infile("../../../um/qual_all.dta"); 
    ofstream outfile("qual_pred.dta");
    double userId, movieId, date, rating;
    int num_neighbors = 20;
    //float rmse = 0;

    for (int i = 0; i < T; i++) {
        if (i % 1000 == 0)
            printf("%d\n", i);

        if (infile >> userId >> movieId >> date >> rating) {
            double r = compute_rating(userId, movieId, user_info, top_q, num_neighbors);
            //rmse += pow((r - rating), 2);
            outfile << userId << " " << movieId << " " << date << " " << r << endl;
        }

    }

    // free memory and close writing files
    free(user_info);
    infile.close();
    outfile.close();

    clock_t end = clock();
    printf("Time: %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);
    //printf("RMSE: %f\n", sqrt(rmse/T));
    return 0;
}
