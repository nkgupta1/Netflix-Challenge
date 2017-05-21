#include "knn.hpp"

void hash_info(string file, vector<double> *hash_map) {
    ifstream infile(file);
    double user, movie, date, rating;

    //for (int i = 0; i < 10000; i++) {
    for (int i = 0; i < N; i++) {
        if (infile >> user >> movie >> date >> rating) {
            hash_map[(int)user].push_back(movie);
            hash_map[(int)user].push_back(rating);
        }
        else {
            break;
        }
    }
}


void get_top_q(vector<double> *hash_map, int *top_q) {
    // 1. compute number of ratings per person
    priority_queue<pair<int, int>> q;
    for (int i = 1; i <= U; i++) {
        q.push(pair<int, int>((int)hash_map[i].size(), i));
    }

    // 2. Grab the top Q ones
    for (int i = 0; i < Q; i++) {
        top_q[i] = q.top().second;
        q.pop();
    }
}


double calc_pearson(int user1, int user2, vector<double> *user1_movies, vector<double> *user2_movies) {
    //1. find movies in common btw user1 and user2
    vector<double> user1_ratings;
    vector<double> user2_ratings;

    int it1 = 0;
    int it2 = 0;

    while ((it1 < (int)(*user1_movies).size()-1) && (it2 < (int)(*user2_movies).size()-1))
    {
        // case 1: common movie
        if ((*user1_movies)[it1] == (*user2_movies)[it2]) {
            user1_ratings.push_back((*user1_movies)[it1+1]);
            user2_ratings.push_back((*user2_movies)[it2+1]);
            it1+=2;
            it2+=2;
        }

        //case 2: it1 > it2. so move it2 forward
        else if ((*user1_movies)[it1] > (*user2_movies)[it2])
            it2+=2;

        //case 3: it1 < it2, so move it1 forward
        else
            it1+=2;
    }

    // 2. calculate covariance, variance of movie ratings
    double pearson;
    // no movies in common
    if (user1_ratings.size() == 0)
        return 0; 

    //otherwise compute pearson coefficient
    double user1_avg = (accumulate(user1_ratings.begin(), user1_ratings.end(), 0.0)) / user1_ratings.size();
    double user2_avg = (accumulate(user2_ratings.begin(), user2_ratings.end(), 0.0)) / user2_ratings.size();
    double cov = 0.0;
    double user1_var = 0.0;
    double user2_var = 0.0;

    for (int i = 0; i < (int)user1_ratings.size(); i++) {
        cov += (user1_ratings[i] - user1_avg) * (user2_ratings[i] - user2_avg);
        user1_var += pow(user1_ratings[i] - user1_avg, 2);
        user2_var += pow(user2_ratings[i] - user2_avg, 2);
    }

    // 3. compute pearson
    // if no variation in user1 or user2 ratings, then pearson is 0 bc no correlation evident
    if (user1_var==0 || user2_var ==0)
        return 0;

    pearson = cov / (sqrt(user1_var * user2_var));

    return pearson;
}


double compute_rating(int userId, int movieId, vector<double> *user_info, int* top_q, int num_neighbors)
{
    // 1. Find all users that have watched movieId. if user2 has watched movieId, compute pearson coefficient
    vector<double> pearsons;
    for (int i = 0; i < Q; i++) {
        if (top_q[i] == userId) 
            continue;
        vector<double> current_list = user_info[top_q[i]];
        //index carries the pointer to the movie rating in current_list for user_info[top_q[i]]
        vector<double>::iterator index = find(current_list.begin(), current_list.end(), movieId);
        if (index != current_list.end()) {
            int index2 = (int) (index - current_list.begin());
            double p = calc_pearson(userId, top_q[i], &user_info[userId], &user_info[top_q[i]]);
            // pearson[i], [i+1], [i+2] = pearson coef, userid, movie rating for movieId
            pearsons.push_back((p+1)/2); //person coef
            pearsons.push_back((double)top_q[i]); //userID
            pearsons.push_back(current_list[index2+1]);  //movie rating
        }
    }

    // if no users have seen movie, just return avg of the user's ratings
    if (pearsons.size() == 0)
        return 0;

    // 2. find k largest pearson coefficients and corresponding users
    priority_queue<pair<double, double>> q;
    int size = min(num_neighbors, (int)pearsons.size()/3);
    for (int i = 0; i < size * 3; i+=3) {
        q.push(pair<double, double>(pearsons[i], pearsons[i+2]));
    }

    vector<double> neighbor_ratings;
    for (int i = 0; i < size; i++) {
        neighbor_ratings.push_back(q.top().second);
        q.pop();
    }

    //3. compute expected rating
    double avg_rating;
    avg_rating = (accumulate(neighbor_ratings.begin(), neighbor_ratings.end(), 0.0)) / neighbor_ratings.size();
    
    return avg_rating;
}


