#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <typeinfo>
#include <numeric>
#include <math.h>
#include <queue>
#include <ctime>
#include <algorithm>
#include <time.h>

using namespace std;


double method1(vector<double> user1_ratings, vector<double> user2_ratings) {

    double sumpr = 0;
    double sum1 = (accumulate(user1_ratings.begin(), user1_ratings.end(), 0.0));
    double sum2 = (accumulate(user2_ratings.begin(), user2_ratings.end(), 0.0));
    double sumsq1 = 0;
    double sumsq2 = 0;
    int size = (int) user1_ratings.size();

    for (int i = 0; i < size; i++) {
        sumpr += user1_ratings[i]*user2_ratings[i];
        sumsq1 += pow(user1_ratings[i], 2);
        sumsq2 += pow(user2_ratings[i], 2);
    }

    double numerator = sumpr - (sum1 * sum2)/size;
    double denom = sqrt((sumsq1 - sum1*sum1/size)*(sumsq2 - sum2*sum2/size));
    double pearson = numerator/denom;

    return pearson;
}

double method2(vector<double> user1_ratings, vector<double> user2_ratings) {

    double user1_avg = (accumulate(user1_ratings.begin(), user1_ratings.end(), 0.0)) / user1_ratings.size();
    double user2_avg = (accumulate(user2_ratings.begin(), user2_ratings.end(), 0.0)) / user2_ratings.size();

    double cov = 0.0;
    double user1_var = 0.0;
    double user2_var = 0.0;

    for (int i = 0; i < (int)user1_ratings.size(); i++) {
        cov += (user1_ratings[i] - user1_avg) * (user2_ratings[i] - user2_avg);
        user1_var += pow(user1_ratings[i] - user1_avg, 2);
        user2_var += pow(user2_ratings[i] - user2_avg, 2);
    }

    double pearson = cov / (sqrt(user1_var * user2_var));
    return pearson;
}


int main () {
    for (int i = 0; i < 4; i++) {
        if (i == 2)
            continue;
        cout << i << endl;
    }

    return 0;
}


//g++ -c -Wall -g -ansi -pedantic -ggdb -std=c++11 -O2 -x c++ test.cpp -o test.o
//g++ -o test test.o