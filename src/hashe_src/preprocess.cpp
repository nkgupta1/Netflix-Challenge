/*!
 * Preprocessing of file. Techniques shamelessly taken from
 * http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/
 */

#include "preprocess.hpp"
#include "util.hpp"

/*
 * We have one giant .dta file; want it in different chunks separated
 * by their group.
 */
void split(const string data, const string ids) {
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
 * We may want to subtract off the mean from the data. This requires
 * finding it first.
 */
float find_average(const string data) {
    // User, month, date, rating
    int u, m, d, r;

    // Data
    ifstream infile(data);

    int sum = 0;
    int count = 0;
    while (infile >> u >> m >> d >> r) {
        if (r != 0) {
            sum += r;
            count++;
        }
    }

    float mean = (float) sum / count;
    return mean;
}

/*
 * Given a file, finds the average deviation of the rating of the movie
 * from the average rating (i.e., MOVIE_AVERAGE - OVERALL_AVERAGE). This
 * can subtracted from all relevant data points so that algorithm can
 * focus on other factors of the score.
 *
 * If no ratings entered, entry is 0; i.e., make no assumptions.
 */
void movie_average(const string data, float (*mov_ave)[M], float ave) {
    // User, month, date, rating, loop counter
    int u, m, d, r, i;

    // Keep track of how many times movie is seen
    int counts[M] = {0};

    // Data
    ifstream infile1(data);
    ifstream infile2(data);

    // Make two passes; first to count number of times a
    // movie is seen, second to compute average. Yes this
    // could be faster, but it's fast enough already.
    while (infile1 >> u >> m >> d >> r) {
        counts[m-1]++;
    }
    while (infile2 >> u >> m >> d >> r) {
        (*mov_ave)[m-1] += (float) r / counts[m-1];
    }

    // We now have the movie averages. Subtract off the overall
    // average.
    for (i = 0; i < M; i++) {
        if ((*mov_ave)[i] != 0) {
            (*mov_ave)[i] -= ave;
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
 */
void user_average(const string data, float (*use_ave)[U], float ave) {
    // User, month, date, rating, loop counter
    int u, m, d, r, i;

    // Keep track of how many movies a user rates. Initialize everything
    // to zero.
    int counts[U] = {0};

    // Data
    ifstream infile1(data);
    ifstream infile2(data);

    // Make two passes; first to count number of times a
    // user is seen, second to compute average. Yes this
    // could be faster, but it's fast enough already.
    while (infile1 >> u >> m >> d >> r) {
        counts[u-1]++;
    }
    while (infile2 >> u >> m >> d >> r) {
        (*use_ave)[u-1] += (float) r / counts[u-1];
    }

    // We now have the movie averages. Subtract off the overall
    // average.
    for (i = 0; i < U; i++) {
        if ((*use_ave)[i] != 0) {
            (*use_ave)[i] -= ave;
        }
    }
}

/*
 * Carries out the nitty-gritty of pre-processing. Reads in the file
 * we want modified, subtracts off the average, user deviation, and movie
 * deviation, and writes it to a new file (= processed_<file> )
 *
 * When more preprocessing techniques are added, they will be added here.
 */
void preprocess(const string data, float (*mov_ave)[M], 
                float (*use_ave)[U], float ave) {
    // User, month, date, rating, id
    int u, m, d;
    float r, new_r;

    // Directory and file names - used for saving new .dta files
    string dir = directory_from_path(data);
    string file = file_from_path(data);

    // Input file
    ifstream datafile(data);

    // Output file
    ofstream processed(dir + "/processed_" + file);

    while (datafile >> u >> m >> d >> r) {
        new_r = r - ave - (*use_ave)[u-1] - (*mov_ave)[m-1];
        processed << u << " " << m << " " << d << " " << new_r << "\n";
    } 
}

/*
 * Given predictions for processed data, adds back in factors to get
 * estimates of actual scores. Essentially unpreprocesses the data.
 *
 * ranking = ranking + ave + user_ave + movie_ave
 */
void unpreprocess(const string data, float (*mov_ave)[M], 
                  float (*use_ave)[U], float ave) {
    // User, month, date, rating, id
    int u, m, d;
    float r, new_r;

    // Directory and file names - used for saving new .dta files
    string dir = directory_from_path(data);
    string file = file_from_path(data);

    // Input file
    ifstream datafile(data);

    // Output file
    ofstream processed(dir + "/unprocessed_" + file);

    while (datafile >> u >> m >> d >> r) {
        new_r = r + ave + (*use_ave)[u-1] + (*mov_ave)[m-1];
        processed << u << " " << m << " " << d << " " << new_r << "\n";
    } 
}

/*
 * Example "do everything function" if you're curious about
 * the syntax. Takes about 6-7 minutes to run on my machine.
 * Generates about 6 gigabytes of files. 
 */
int main() {
    split("../data/um/all.dta", "../data/um/all.idx");
    printf("Split\n");

    float ave = find_average("../data/um/base_all.dta");
    printf("Average: %f\n", ave);

    float use_ave[U];
    user_average("../data/um/base_all.dta", &use_ave, ave);
    // printf("%f\n", use_ave[0]);
    printf("User averages found\n");

    float mov_ave[M];
    movie_average("../data/um/base_all.dta", &mov_ave, ave);
    // printf("%f\n", mov_ave[78]);
    printf("Movie averages found\n");

    preprocess("../data/um/base_all.dta", &mov_ave, &use_ave, ave);
    printf("Processed\n");

    unpreprocess("../data/um/processed_base_all.dta", &mov_ave, 
        &use_ave, ave);
    printf("Unprocessed\n");

    return 0;
}