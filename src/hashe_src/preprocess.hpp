#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <stdexcept>

// #define U   458293      // Users
#define U   23          // Users
#define M   17770       // Movies
// #define N   102416306   // Data Points
#define N   4552        // Data Points in function testing set
#define D   2243        // Number days
#define K   5           // Number ratings

using namespace std;

/* 
 * Pre-process class. Specifies the parameters changed during pre-processing.
 * Created and saved to a file when preprocess is run, requested by 
 * unpreprocess.
 *
 * ONLY USED INTERNALLY TO PREPROCESSING SCRIPT; OTHER CODE SHOULD NOT
 * REQUIRE EXISTING PROCESSING OBJECT
 */
class Processing {
    // VARIABLES

    // Average over all data
    float ave;

    // User average, stdev;
    float use_ave[U];
    float use_stdev[U];

    // Movie average
    float mov_ave[M];

    // Keep track of the file from which this data was generated;
    string file;

public:
    // PUBLIC FUNCTIONS

    // Constructor, Destructor
    Processing();
    ~Processing();

    // Split data; takes in data file and index file.
    // Splits into 5 new files (base, valid, hidden, probe, qual)
    static void split(const string data, const string ids);

    // Sets parameters according to data in given file, saves to outname.
    void initialize(const string data, const string outname);

    // Preprocess a file, saving the modified data in outname
    void preprocess(const string infile, const string outfile);

    // Given a file trained on processed weights, map back
    void unpreprocess(const string infile, const string outfile);

    // Methods for saving, loading
    void save(const string fname);
    void load(const string fname);

private:
    // PRIVATE FUNCTIONS

    // Methods for finding averages. At some point, could probably
    // consolidate into one function. Called by initialize.
    void global_average();
    void movie_average();
    void user_average();
};

// TODO
// Add in parameters for filenames. Would be easier than the current
// "append" nonsense.

// TODO
// Some of these functions are pretty slow. If anyone wants to learn
// more about c++ I/O and make them faster, please go on right ahead.

// TODO
// Can probably put all averages together in one I/O cycle.

#endif
