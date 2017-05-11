#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define U   458293      // Users
// #define U   23          // Users in my function testing set
#define M   17770       // Movies
#define N   102416306   // Data Points
#define N   4552        // Data Points in function testing set
#define D   2243        // Number days
#define K   5           // Number ratings

using namespace std;

// TODO
// Add in parameters for filenames. Would be easier than the current
// "append" nonsense.

// TODO
// Some of these functions are pretty slow. If anyone wants to learn
// more about c++ I/O and make them faster, please go on right ahead.

// Splits data into 5 new files (base, valid, hidden, probe, qual)
void split(const string data, const string ids);

// Finds average of all data points in a given file
float find_average(const string data);

// Find movie-specific averages
void movie_average(const string data, float (*mov_ave)[M], float ave);

// Find user-specific averages
void user_average(const string data, float (*use_ave)[U], float ave);

// Actually carries out preprocessing on a file.
// This function will expand once more sophisticated methods are added.
void preprocess(const string data, float (*mov_ave)[M], 
                float (*use_ave)[U], float ave);

// Given (labelled) predictions, undoes effects of preprocessing.
void unpreprocess(const string data, float (*mov_ave)[M], 
                  float (*use_ave)[U], float ave);

#endif
