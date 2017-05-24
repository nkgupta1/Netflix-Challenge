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
 * Function found on stack exchange:
 * stackoverflow.com/questions/17925051/fast-textfile-reading-in-c
 * 
 * Given a file, counts number of lines really quickly
 * (2x speed up over naive method)
 */
int line_count(const string file) {
    const char *fname = file.c_str();
    static const auto BUFFER_SIZE = 64*1024;
    int fd = open(fname, O_RDONLY);
    if(fd == -1) {
        return -1;
    }

    /* Advise the kernel of our access pattern.  */
    posix_fadvise(fd, 0, 0, 1);  // FDADVICE_SEQUENTIAL

    char buf[BUFFER_SIZE + 1];
    int lines = 0;

    while(size_t bytes_read = read(fd, buf, BUFFER_SIZE))
    {
        if(bytes_read == (size_t)-1) {
            return -1;
        }
        if (!bytes_read) {
            break;
        }

        for(char *p = buf; 
            (p = (char*) memchr(p, '\n', (buf + bytes_read) - p)); 
            ++p) {
            
            ++lines;
        }
    }

    return lines;
}

float sig(float num) {
    return (1.0 / (1.0 + exp(-num)));
}
