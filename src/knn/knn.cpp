#include "knn.hpp"

/* Writes hash_map to binary file 
*/
void to_binary(string file, vector<float> *hash_map) {
    ofstream out(file, ios::binary);
    int size[U+1];
    for (int i = 0; i < U+1; i++) {
        size[i] = hash_map[i].size();
        out.write((char*)&size[i], sizeof(int));
        out.write((char*)(&hash_map[i]), size[i] * sizeof(float));
    }
    out.close();
}


/* Reads data from binary to hash_map 
*/
void from_binary(string file, vector<float> *hash_map) {
    ifstream in(file, ios::binary);
    int size[U+1];
    for (int i = 0; i < U+1; i++) {
        in.read((char*)(&size[i]), sizeof(int));
        in.read((char*)(&hash_map[i]), size[i] * sizeof(float));
    }
    in.close();
}


/* Reads data in as array of vectors where array is indexed by user id.
   hash_map[userId] = vector which alternates movieId and rating for movieId. so
   hash_map[userId][i] = movieId if i even, = rating for movieId at i-1 when odd
*/
void hash_info(string file, vector<float> *hash_map) {
    ifstream infile(file);
    float user, movie, date, rating;

    //for (int i = 0; i < N; i++) {
    for (int i = 0; i < 1000; i++) {
        if (infile >> user >> movie >> date >> rating) {
            hash_map[(int)user].push_back(movie);
            hash_map[(int)user].push_back(rating);
        }
        else {
            break;
        }
    }
}

/* Gets the top Q users with the most ratings
*/
void get_top_q(vector<float> *hash_map, int *top_q) {
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

/* Calculate the pearson coefficient between two users 
   (essentially the covariance level)
*/
float calc_pearson(int user1, int user2, vector<float> *user1_movies, vector<float> *user2_movies) {
    // 1. find movies in common btw user1 and user2
    vector<float> user1_ratings;
    vector<float> user2_ratings;
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

        // case 2: it1 > it2. so move it2 forward
        else if ((*user1_movies)[it1] > (*user2_movies)[it2])
            it2+=2;

        // case 3: it1 < it2, so move it1 forward
        else
            it1+=2;
    }

    // 2. calculate covariance, variance of movie ratings
    float pearson;
    float user1_avg = (accumulate(user1_ratings.begin(), user1_ratings.end(), 0.0)) / user1_ratings.size();
    float user2_avg = (accumulate(user2_ratings.begin(), user2_ratings.end(), 0.0)) / user2_ratings.size();
    float cov = 0.0;
    float user1_var = 0.0;
    float user2_var = 0.0;

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

/* Compute rating for userId and movieId by finding most similar num_neighbors users 
   (highest pearson coef) that also watched movieId and compute average of those ratings.
*/
float compute_rating(int userId, int movieId, vector<float> *user_info, int* top_q, int num_neighbors, MyMap &userId_map)
{
    // 1. Find all users that have watched movieId. if user2 has watched movieId, compute pearson coefficient
    vector<float> pearsons;
    float p;
    for (int i = 0; i < Q; i++) {

        // if userId is already part of top_q, ignore it
        if (top_q[i] == userId) 
            continue;

        vector<float> current_list = user_info[top_q[i]];
        long int combined = ((long)userId << 32) + top_q[i];

        //index carries the pointer to the movie rating in current_list for user_info[top_q[i]]
        vector<float>::iterator index = find(current_list.begin(), current_list.end(), movieId);
        if (index != current_list.end()) {
            int index2 = (int) (index - current_list.begin());

            // check hash map to see if we already computed pearson coefficient
            MyMap::iterator got = userId_map.find(combined);
            if (got != userId_map.end())
               p = got->second;

            // if not saved, then compute it
            else {
               p = calc_pearson(userId, top_q[i], &user_info[userId], &user_info[top_q[i]]);
               userId_map[combined] = p;
            }
            pearsons.push_back(p); // pearson coeff
            pearsons.push_back(current_list[index2 + 1]); //movie rating
        }
    }

    // if no users have seen movie, just return avg of the user's ratings
    if (pearsons.size() == 0)
        return 0;

    // 2. find k largest pearson coefficients and corresponding users
    priority_queue<pair<float, float>> q;
    for (int i = 0; i < (int)pearsons.size(); i+= 2) {
        q.push(pair<float, float>(pearsons[i], pearsons[i+1]));
    }

    vector<float> neighbor_ratings;
    for (int i = 0; i < min(num_neighbors, (int)pearsons.size()/2); i++) {
        neighbor_ratings.push_back(q.top().second);
        q.pop();
    }

    // 3. compute expected rating
    float avg_rating;
    avg_rating = (accumulate(neighbor_ratings.begin(), neighbor_ratings.end(), 0.0)) / neighbor_ratings.size();
    
    return avg_rating;
}


