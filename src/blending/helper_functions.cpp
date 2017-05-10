//
//  read_data
//
//  Created by Skim on 5/8/17.
//  Copyright © 2017 Seohyun Kim. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;



vector<vector<float>> get_ratings(vector<string> filenames, int num_points) {
    vector<vector<int>> input_info(num_points);
    vector<vector<float>> rating_data(num_points);
    int user, movie, date;
    float rating;
    
    // get initial info
    cout << "First filename " << filenames[0] << endl;
    ifstream infile(filenames[0]);
    int i = 0;
    while (infile >> user >> movie >> date >> rating) {
        rating_data[i].push_back(rating);
        input_info[i] = {user, movie, date};
        i++;
    }
    
    for (int j = 1; j < filenames.size(); j ++) {
        ifstream infile(filenames[j]);
        int i = 0;
        while (infile >> user >> movie >> date >> rating) {
            if (input_info[i][0] != user || input_info[i][1] != movie || input_info[i][2] != date) {
                cerr << "Input user/ movie/ date does not match\n";
                exit(1);
            }
            rating_data[i].push_back(rating);
            i++;
        }
    }
    return rating_data;
}

