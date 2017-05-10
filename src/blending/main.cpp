//
//  main.cpp
//
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include "helper_functions.hpp"
using namespace std;


// command line argument: <string of qual.dta files delimited by space> <rmse delimited by space>
int main (int argc, char *argv[]) {
    int num_points = 10;

    // read in the ratings from input
    istringstream iss2(argv[1]);
    vector<string> filenames;
    string i2;
    while (iss2 >> i2) {
        filenames.push_back(i2);
    }
    vector<vector<float>> ratings(num_points);
    ratings = get_ratings(filenames, num_points);
    
    
    // read rmse from input
    istringstream iss(argv[2]);
    vector<float> rmse;
    float i;
    while (iss >> i) {
        rmse.push_back(i);
    }
    float rmse_sum = accumulate(rmse.begin(), rmse.end(), 0.0);

    
    // get the rmse ratios
    int num_methods = static_cast<unsigned int>(rmse.size());
    vector<float> rmse_ratio(num_methods);
    for (int i = 0; i < num_methods; i++) {
        rmse_ratio[i] = rmse_sum / rmse[i];
    }
    float normalization = accumulate(rmse_ratio.begin(), rmse_ratio.end(), 0.0);
    
    
    // update the results by simple blending
    vector<float> result(num_points, 0.0);
    for (int i = 0; i < num_points; i ++) {
        for (int j = 0; j < num_methods; j ++) {
            result[i] += ratings[i][j]*rmse_ratio[j];
        }
        result[i] /= normalization;
    }

    
    // print results
    for (int i = 0; i < num_points; i++) {
        cout <<i << ": " << result[i]<< endl;
    }

    return 0;

    
}



