#include "knn.hpp"

void hash_info(string file, vector<double> (*hash_map))
{
    double current_user = 1;
    ifstream infile(file);
    vector<double> current_map;
    double user, movie, date, rating;
    for (int k=0; k < N*4; k+=4)
    {
        if (infile >> user >> movie >> date >> rating)
        {
            if (k%4000000 == 0)
            {
                cout << k/4 << endl;
            }

            if (user == current_user)
            {
                current_map.push_back(movie);
                current_map.push_back(rating);
            }

            else
            {
                (hash_map)[(int)current_user]=current_map;
                current_user = user;
                current_map.clear();
                current_map.push_back(movie);
                current_map.push_back(rating);
            }
        }

        else
        {
            break;
        }
        (hash_map)[(int)current_user] = current_map;
    }
};

void get_top_q(vector<double> (*hash_map), int (*top_q) [Q])
{
    // 1. compute number of ratings per person
    int user_num_ratings[U] = {0};
    for (int j=1; j<=U; j+=1)
    {
        user_num_ratings[j] = (hash_map)[j].size();
    }

    priority_queue<vector<int>> q;
    for (int i = 1; i <= Q; ++i)
    {
        q.push({user_num_ratings[i], i});
    }

    for (int k=0; k< Q; ++k)
    {
        (*top_q)[k] = q.top()[1];
        q.pop();
    }


}

double calc_pearson(int user1, int user2, vector<double> *user1_movies, vector<double> *user2_movies)
{
    //1. find movies in common btw user1 and user2
    int n = 0;
    vector<double> user1_ratings;
    vector<double> user2_ratings;

    int it1 = 0;
    int it2 = 0;

    while (it1 < (*user1_movies).size()-1 & it2 < (*user2_movies).size()-1)
    {
        // case 1: common movie
        if ((*user1_movies)[it1] == (*user2_movies)[it2])
        {
            user1_ratings.push_back((*user1_movies)[it1+1]);
            user2_ratings.push_back((*user2_movies)[it2+1]);
            it1+=2;
            it2+=2;
        }

            //case 2: it1 > it2. so move it2 forward
        else if ((*user1_movies)[it1] > (*user2_movies)[it2])
        {
            it2+=2;
        }

            //case 3: it1 < it2, so move it1 forward
        else
        {
            it1+=2;
        }
    }

    /*  clock_t end2 = clock();
      cout<<(end2 - begin2)<<endl;

      this_thread::sleep_for(chrono::milliseconds(100));*/
    // 2. calculate covariance, variance of movie ratings

    double pearson;
    //  clock_t begin3 = clock();
    // no movies in common
    if (user1_ratings.size()==0)
    {
        pearson = 0;
    }

        //otherwise compute pearson coefficient
    else {
        double user1_avg = (accumulate(user1_ratings.begin(), user1_ratings.end(), 0.0)) / user1_ratings.size();
        double user2_avg = (accumulate(user2_ratings.begin(), user2_ratings.end(), 0.0)) / user2_ratings.size();

        double cov = 0.0;
        double user1_var = 0.0;
        double user2_var = 0.0;

        for (int i = 0; i < user1_ratings.size(); i++) {
            cov += (user1_ratings[i] - user1_avg) * (user2_ratings[i] - user2_avg);
            user1_var += pow(user1_ratings[i] - user1_avg, 2);
            user2_var += pow(user2_ratings[i] - user2_avg, 2);

        }
        // 3. compute pearson

        // if no variation in user1 or user2 ratings, then pearson is 0 bc no correlation evident
        if (user1_var==0 || user2_var ==0)
        {
            pearson = 0;
        }

        else
        {
            pearson = cov / (sqrt(user1_var * user2_var));
        }

    }
    /*   clock_t end3 = clock();
       cout<<(end3-begin)<<endl;
       cout<<"end"<<endl;
       this_thread::sleep_for(chrono::milliseconds(100));*/

    return pearson;
}

// was used when pearson was precomputed
/*
void hash_pearson( string pearsonsfile, vector<double>* users_pearson)
{
    // 1. read file
    ifstream pearsons(pearsonsfile);
    double user1, user2, pearson_coef;
    while (pearsons >> user1 >> user2 >> pearson_coef)
    {
        (users_pearson)[(int)user1].push_back(user2);
        (users_pearson)[(int)user1].push_back(pearson_coef);
    }
};
*/

double compute_rating(int userId, int movieId, vector<double> *user_info, int* top_q, int num_neighbors)
{
    user_info[userId] = {}; //erase itself from consideration

    // 1. Find all users that have watched movieId. if user2 has watched movieId, compute pearson coefficient
    vector<double> pearsons;
    for (int i=0; i<Q; i++)
    {
        vector<double> current_list = user_info[top_q[i]];

        //index carries the index of the movie rating in current_list for user_info[top_q[i]]
        vector<double>::iterator index = find(current_list.begin(), current_list.end(), movieId);
        if (index != current_list.end())
        {
            ptrdiff_t index2 = index-current_list.begin();
            double p = calc_pearson(userId, top_q[i], &(user_info)[(int)userId], &(user_info)[top_q[i]]);
            // pearson[i], [i+1], [i+2] = pearson coef, userid, movie rating for movieId
            pearsons.push_back((p+1)/2);
            pearsons.push_back((double)top_q[i]);
            pearsons.push_back(current_list[(int)(index2+1)]);
        }
    }


    // 2. find k largest pearson coefficients and corresponding users
    priority_queue<pair<double, double>> q;
    for (int i = 0; i < pearsons.size(); i+=3)
    {
        q.push(pair<double, double>(pearsons[i], pearsons[i+2]));
    }

    vector<double> neighbor_ratings;
    for (int i = 1; i <= pearsons.size()/3; ++i)
    {
        double neighbor_rate = q.top().second;
        neighbor_ratings.push_back(neighbor_rate);
        q.pop();
    }

    //3. compute expected rating

    double avg_rating;
    //if no users have seen movie, just return avg of the user's ratings
    if (pearsons.size() == 0)
    {
        avg_rating = 0; //normalized
    }
        //otherwise, return the average of num_neighbors rating
    else
    {
        avg_rating = (accumulate(neighbor_ratings.begin(), neighbor_ratings.end(), 0.0)) / neighbor_ratings.size();
    }
    return avg_rating;
}

