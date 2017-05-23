// #include "util.hpp"
//#include "preprocess.hpp"
#include "knn.hpp"
#include <time.h>

int main () {
    clock_t begin = clock();

    // 1. Read in data
    printf("Hashing data\n");
    vector<float> *user_info;
    user_info = (vector<float> *)malloc((U+1) * sizeof(vector<float>));
    hash_info("../../../um/base_all.dta", user_info);
    //to_binary("numbers.dta", user_info);
    //from_binary("numbers.dta", user_info2);

    // 2. Get Q users with highest number of ratings
    printf("Getting top q users\n");
    int top_q[Q];
    get_top_q(user_info, top_q);


    // 3. Compute ratings for infile points
    ifstream infile("../../../um/qual_all.dta"); 
    ofstream outfile("qual_pred.dta");
    float userId, movieId, date, rating;
    int num_neighbors = 20;
    //float rmse = 0;
    float curr_userId = 1;
    MyMap userId_map;

    for (int i = 0; i < T; i++) {
        if (i % 10000 == 0)
            printf("%d\n", i);

        if (infile >> userId >> movieId >> date >> rating) {
            if (userId != curr_userId) {
                curr_userId = userId;
                userId_map.clear();
            }
            float r = compute_rating(userId, movieId, user_info, top_q, num_neighbors, userId_map);
            //rmse += pow((r - rating), 2);
            outfile << userId << " " << movieId << " " << date << " " << r << endl;
        }
        else {
            break;
        }

    }

    // 4. free memory and close writing files
    free(user_info);
    infile.close();
    outfile.close();

    clock_t end = clock();
    printf("Time: %f seconds\n", (float)(end - begin)/CLOCKS_PER_SEC);
    //printf("RMSE: %f\n", sqrt(rmse/T));
    return 0;
}
