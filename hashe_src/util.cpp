#include "util.hpp" 

const string directory_from_path(const string str) {
    int found;

    found = str.find_last_of("/\\");
    string dir = str.substr(0,found);
    return dir;
}

const string file_from_path(const string str) {
    int found;

    found = str.find_last_of("/\\");
    string file = str.substr(found + 1);
    return file;
}

/*
 * Reads in file data into provided array, with known
 * number of rows and cols.
 *
 * Figure out a way to make this faster at some point?
 */

// Lol, can't read in whole file. Sucks to suck, rip.
// void read_file(const char* fname, float arr[][4], int rows) {
//     ifstream infile(fname);
//     float u, m, d, r;
//     int i;
//     char c;

//     for (i = 0; i < rows; i++) {
//         if ((infile >> u >> c >> m >> c >> d >> c >> r) && (c == ',')) {
//             arr[i][0] = u;
//             arr[i][1] = m;
//             arr[i][2] = d;
//             arr[i][3] = r;
//         }
//         else {
//             break;
//         }
//     }
// }

// int main() {
//     printf("Is it here?\n");
//     float arr[L][4];
//     printf("Or here\n");
//     read_file("../hashe_data/um/test.dta", arr, L);

//     return 0;
// }

// int main() {
//     return 1;
// }