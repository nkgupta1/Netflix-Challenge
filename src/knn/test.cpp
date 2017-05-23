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
#include <unordered_map>
#include "knn.hpp"

using namespace std;


void print_binary(long int n) {
    while (n) {
    if (n & 1)
        printf("1");
    else
        printf("0");
    n >>= 1;
    }
    printf("\n");
}

int main () {
    ifstream infile("qual_pred.dta"); 
    ofstream outfile("qual_pred_format.dta");
    float userId, movieId, date, rating;

    for (int i = 0; i < T; i++) {
        if (infile >> userId >> movieId >> date >> rating) {
            outfile << rating << endl;
        }
    }
        


    return 0;
}


//g++ -c -Wall -g -ansi -pedantic -ggdb -std=c++11 -O2 -x c++ test.cpp -o test.o
//g++ -o test test.o