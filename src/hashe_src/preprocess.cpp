/*!
 * Preprocessing of file. Techniques shamelessly taken from
 * http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
 */

#include "preprocess.hpp"
#include "util.hpp"

/*
 * Constructor for Processing object.
 */
Processing::Processing() {
    ave = 0;
    memset(use_ave, 0, sizeof(use_ave));
    memset(use_stdev, 0, sizeof(use_stdev));
    memset(mov_ave, 0, sizeof(mov_ave));
    file = "null";
}

/*
 * Destructor for Processing object.
 */
Processing::~Processing() {
    // Nothing to deallocate
}

/*
 * We have one giant .dta file; want it in different chunks separated
 * by their group.
 */
void Processing::split(const string data, const string ids) {
    // User, month, date, rating, id
    int u, m, d, r, id;

    // Directory and file names - used for saving new .dta files
    string dir = directory_from_path(data);
    string file = file_from_path(data);

    // Input files
    ifstream datafile(data);
    ifstream idsfile(ids);

    // Output files
    ofstream base(dir + "/base_" + file);
    ofstream valid(dir + "/valid_" + file);
    ofstream hidden(dir + "/hidden_" + file);
    ofstream probe(dir + "/probe_" + file);
    ofstream qual(dir + "/qual_" + file);

    while (datafile >> u >> m >> d >> r) {
        idsfile >> id;
        if        (id == 1) {
            base << u << " " << m << " " << d << " " << r << "\n";
        } else if (id == 2) {
            valid << u << " " << m << " " << d << " " << r << "\n";
        } else if (id == 3) {
            hidden << u << " " << m << " " << d << " " << r << "\n";
        } else if (id == 4) {
            probe << u << " " << m << " " << d << " " << r << "\n";
        } else if (id == 5) {
            qual << u << " " << m << " " << d << " " << r << "\n";
        }
    } 
}

/*
 * Generate and save pre-processing object. Will write more helpful
 * comment later.
 */
void Processing::initialize(const string data, const string outname) {
    // Save the data file for future reference
    file = data;

    // Generate data
    global_average();
    user_average();
    movie_average();

    // Save to file
    save(outname);
}

/*
 * Carries out the nitty-gritty of pre-processing, no questions asked. 
 * Reads in the file we want modified, subtracts off the average, user 
 * deviation, and movie deviation, divides by user-stdev, 
 * and writes it to a new file (= processed_<file> ).
 *
 * When more preprocessing techniques are added, they will be added here.
 */
void Processing::preprocess(const string infile, const string outfile) {
    // Make sure Processing object is initialize
    if (file == "null") {
        throw runtime_error("Processing object not initialized. Call "
            "initialize or load before this method");
    }

    // User, month, date, rating, id
    int u, m, d;
    float r, new_r;

    // Input file
    ifstream datafile(infile);

    // Output file
    ofstream processed(outfile);

    while (datafile >> u >> m >> d >> r) {
        new_r = (r - ave - use_ave[u-1] - mov_ave[m-1]) / use_stdev[u-1];
        processed << u << " " << m << " " << d << " " << new_r << "\n";
    }
}

/*
 * Given predictions for processed data, adds back in factors to get
 * estimates of actual scores. Essentially unpreprocesses the data.
 *
 * ranking = ranking + ave + user_ave + movie_ave
 */
void Processing::unpreprocess(const string infile, const string outfile) { 
    // Make sure Processing object is initialize
    if (file == "null") {
        throw runtime_error("Processing object not initialized. Call "
            "initialize or load before this method");
    }

    // User, month, date, rating, id
    int u, m, d;
    float r, new_r;

    // Input file
    ifstream datafile(infile);

    // Output file
    ofstream processed(outfile);

    while (datafile >> u >> m >> d >> r) {
        new_r = (r * use_stdev[u-1]) + ave + use_ave[u-1] + mov_ave[m-1];
        processed << u << " " << m << " " << d << " " << new_r << "\n";
    }
}

/*
 * Save to file for processing data
 */
void Processing::save(const string fname) {
    int i;
    ofstream ofs(fname);

    // Save average
    ofs << ave << " ";

    // Save each user average, stdev
    for (i = 0; i < U; i++) {
        ofs << use_ave[i] << " " << use_stdev[i] << " ";
    }

    // Save each movie average
    for (i = 0; i < M; i++) {
        ofs << mov_ave[i] << " ";
    }

    // Save file
    ofs << file << " ";
}

/*
 * Load from file for processing data
 */
void Processing::load(const string fname) {
    int i;
    ifstream ifs(fname);

    // Load average
    ifs >> ave;

    // Load each user average, stdev
    for (i = 0; i < U; i++) {
        ifs >> use_ave[i] >> use_stdev[i];
    }

    // Load each movie average
    for (i = 0; i < M; i++) {
        ifs >> mov_ave[i];
    }

    // Load file
    ifs >> file;
}

/*
 * We may want to subtract off the mean from the data. This requires
 * finding it first.
 */
void Processing::global_average() {
    // User, month, date, rating
    int u, m, d, r;

    // Data
    ifstream infile(file);

    int sum = 0;
    int count = 0;
    while (infile >> u >> m >> d >> r) {
        if (r != 0) {
            sum += r;
            count++;
        }
    }

    ave = (float) sum / count;
}

/*
 * Given a file, finds the average deviation of the rating of the movie
 * from the average rating (i.e., MOVIE_AVERAGE - OVERALL_AVERAGE). This
 * can subtracted from all relevant data points so that algorithm can
 * focus on other factors of the score.
 *
 * If no ratings entered, entry is 0; i.e., make no assumptions.
 */
void Processing::movie_average() {
    // User, month, date, rating, loop counter
    int u, m, d, r, i;

    // Keep track of how many times movie is seen
    int counts[M] = {0};

    // Data
    ifstream infile1(file);
    ifstream infile2(file);

    // Make two passes; first to count number of times a
    // movie is seen, second to compute average. Yes this
    // could be faster, but it's fast enough already.
    while (infile1 >> u >> m >> d >> r) {
        counts[m-1]++;
    }
    while (infile2 >> u >> m >> d >> r) {
        mov_ave[m-1] += (float) r / counts[m-1];
    }

    // We now have the movie averages. Subtract off the overall
    // average.
    for (i = 0; i < M; i++) {
        if (mov_ave[i] != 0) {
            mov_ave[i] -= ave;
        }
    }
}

/*
 * Given a file, finds the average deviation of the ratings of the user
 * from the average rating (i.e., USER_AVERAGE - OVERALL_AVERAGE). This
 * can subtracted from all relevant data points so that algorithm can
 * focus on other factors of the score.
 *
 * If no ratings entered, entry is 0; i.e., make no assumptions.
 *
 * Also determines user standard deviation; this is useful for 
 * determining a users normal "range" of ratings.
 */
void Processing::user_average() {
    // User, month, date, rating, loop counter
    int u, m, d, r, i;

    // Keep track of how many movies a user rates. Initialize everything
    // to zero.
    int counts[U] = {0};

    // Data
    ifstream infile1(file);
    ifstream infile2(file);
    ifstream infile3(file);

    // Make three passes; first to count number of times a
    // user is seen, second to compute average, third for
    // stdev. Yes this could be faster, but it's fast enough 
    // already.
    while (infile1 >> u >> m >> d >> r) {
        counts[u-1]++;
    }
    while (infile2 >> u >> m >> d >> r) {
        use_ave[u-1] += (float) r / counts[u-1];
    }
    while (infile3 >> u >> m >> d >> r) {
        use_stdev[u-1] += pow((float) r - use_ave[u-1], (float) 2) 
                            / (counts[u-1] - 1);
    }

    // We now have the movie averages. Subtract off the overall
    // average.
    for (i = 0; i < U; i++) {
        if (use_ave[i] != 0) {
            use_ave[i] -= ave;
        }
    }

    // Also find stdev = sqrt(variance)
    for (i = 0; i < U; i++) {
        if (use_stdev[i] != 0) {
            use_stdev[i] = sqrt(use_stdev[i]);
        }
    }
}

/*
 * Example "do everything function" if you're curious about
 * the syntax. Takes about 6-7 minutes to run on my machine.
 * Generates about 6 gigabytes of files. 
 */
int main() {
    // Processing::split("hashe_data/um/test.dta", "hashe_data/um/test.idx");
    // printf("Split\n");

    // float ave = find_average("hashe_data/um/base_test.dta");
    // printf("Average: %f\n", ave);

    // float use_ave[U];
    // float use_stdev[U];
    // user_average("hashe_data/um/base_test.dta", &use_ave, 
    //     &use_stdev, ave);
    // printf("%f\n", use_ave[0] + ave);
    // printf("User averages found\n");

    // float mov_ave[M];
    // movie_average("hashe_data/um/base_test.dta", &mov_ave, ave);
    // // printf("%f\n", mov_ave[78]);
    // printf("Movie averages found\n");

    // preprocess("hashe_data/um/base_test.dta");
    // printf("Processed\n");

    // unpreprocess("hashe_data/um/processed_base_test.dta");
    // printf("Unprocessed\n");

    // map_preprocessing("hashe_data/um/base_test.dta", "hashe_data/um/test.txt");
    // Processing test;
    // test.load("hashe_data/um/test.txt");
    // test.save("hashe_data/um/test2.txt");
    Processing test;
    test.initialize("hashe_data/um/base_test.dta", "hashe_data/um/test.txt");
    test.preprocess("hashe_data/um/base_test.dta", "hashe_data/um/pro_base.dta");
    test.unpreprocess("hashe_data/um/pro_base.dta", "hashe_data/um/unpro_base.dta");
    return 0;
}