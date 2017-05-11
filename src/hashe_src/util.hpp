#ifndef __UTIL_H__
#define __UTIL_H__

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

// Get parts of file names
const string directory_from_path(const string str);
const string file_from_path(const string str);

// Memcopy function would be nice, so long as there's no vectory nonsense

#endif