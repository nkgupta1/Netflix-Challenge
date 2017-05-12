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
