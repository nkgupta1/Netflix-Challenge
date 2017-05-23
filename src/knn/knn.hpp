//
// Created by Lilly Luo on 5/12/17.
//

#ifndef TRY1_KNN_HPP
#define TRY1_KNN_HPP

//TODO Import and preprocess hidden file as validation

#define U   458293      // Number of Users
//#define U   23          // Users in my function testing set

#define M   17770       // Number of Movies
//#define N    4000000  // Testing stuff
#define N   94362233        // Data Points in base
#define T   2749898    // Data points in qual set (hidden)
#define D   2243        // Number days
#define K   5           // Number ratings
#define Q   500        // Top Q optimization - only consider ratings of top Q ppl as neighbors

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

#include <chrono>
#include <thread>

using namespace std;

typedef unordered_map<long int, float> MyMap;

void to_binary(string file, vector<float> *hash_map);
void from_binary(string file, vector<float> *hash_map);

void hash_info(string file, vector<float> (*hash_map));
void get_top_q(vector<float> (*hash_map), int (*top_q));

float calc_pearson(int user1, int user2, vector<float> *user1_movies, vector<float> *user2_movies);
float compute_rating(int userId, int movieId, vector<float> *user_info, int* top_q, int num_neighbors, MyMap &userId_map);

#endif //TRY1_KNN_HPP
