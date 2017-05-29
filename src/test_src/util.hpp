#ifndef __UTIL_H__
#define __UTIL_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>


#include <fcntl.h>
#include <unistd.h>

#define U   458293      // Users
#define M   17770       // Movies
// #define D   2243        // Number days

using namespace std;

// Get parts of file names
const string directory_from_path(const string str);
const string file_from_path(const string str);

// Line count; useful if you don't want to just hardcode these numbers
int line_count(const string file);

// General sigmoid function
float sig(float num);

#endif