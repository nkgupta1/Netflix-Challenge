//
// Created by Lilly Luo on 5/12/17.
//

#ifndef TRY1_KNN_HPP
#define TRY1_KNN_HPP

//TODO Import and preprocess hidden file as validation

#define U   458293      // Number of Users
//#define U   23          // Users in my function testing set

#define M   17770       // Number of Movies
// #define N   102416306   // Data Points
//#define N    4000000  // Testing stuff
#define N   94362233        // Data Points in base
#define T   1964391    // Data points in test set (hidden)
//#define T   100
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

#include <chrono>
#include <thread>

using namespace std;

//void read_data(string file, double (*all_data));

// reads data in as array of vectors where array is indexed by user id.
// hash_map[userId] = vector which alternates movieId and rating for movieId. so
// hash_map[userId][i] = movieId if i even, = rating for movieId at i-1 when odd
void hash_info(string file, vector<double> (*hash_map));
//void hash_info( double (*data_table), vector<double> (*hash_map));

// gets the top Q users with the most ratings
void get_top_q(vector<double> (*hash_map), int (*top_q));

// calculate the pearson coefficient between two users (essentially the covariance level)
//double calc_pearson(int user1, int user2, vector<double> *user1_movies, vector<double> *user2_movies);
double calc_pearson(int user1, int user2, vector<double> *user1_movies, vector<double> *user2_movies);
//double calc_pearson(int user1, int user2, double *user1_movies, double *user2_movies);



//void hash_pearson( string pearsonsfile, vector<double>* users_pearson);

// compute rating for userId and movieId by finding most similar num_neighbors users (highest pearson coef)
// that also watched movieId and compute average of those ratings.
double compute_rating(int userId, int movieId, vector<double> *user_info, int* top_q, int num_neighbors);



#endif //TRY1_KNN_HPP