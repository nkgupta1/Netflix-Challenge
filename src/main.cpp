// #include "util.hpp"
//#include "preprocess.hpp"
#include "knn.hpp"

int main (){
    cout << N << endl;

    //1. Read in data
    cout << "Hashing" << endl;
    vector<double>* user_info = new vector<double>[U];
    hash_info("../data/um/processed_base_all.dta", user_info);

    //2. Get Q users with highest number of ratings
    cout << "Getting top Q" << endl;
    int top_q[Q];
    get_top_q(user_info, &top_q);

// *ignore* previously used to precompute pearson coefficient
//    double *test_pred = new double[T];
 //   ofstream pred("../data/um/unprocessed_hidden_preds.dta");
/*    ofstream outfile("../data/pearson_base.dta");
    cout << "Calculating pearson" << endl;
    for (int userId=1; userId<=U; userId++)
    {
        if (userId%10000==0)
        {
            cout << userId <<endl;
        }


        double norm_pred = compute_rating(user, movie, &user_info, num_neighbors, top_q);
        pred << user << " " << movie << " " << date << " " <<norm_pred << endl;

        //compute pearson for each userId and top_q and read into file
        for (int i=0; i<Q; i++)
        {
            double p = calc_pearson(userId, top_q[i], &(user_info)[(int)userId], &(user_info)[top_q[i]]);
            outfile << userId << " " << top_q[i] << " " << p << endl;
        }
    }*/

 //   outfile.close();
    /*  vector<double>* pearson_info = new vector<double>[U];
  hash_pearson("../data/hidden_base_pearson.dta", pearson_info);
*/

    // compute ratings for infile points
  ifstream infile("../data/um/hidden_all.dta");
  ofstream outfile("../data/um/hidden_pred.dta");
  double userId, movieId, date, rating;
  int num_neighbors = 20;

  for (int i=0; i<T; i++)
  {
      if (i % 100000 == 0) {
          cout << i << endl;
      }

      if (infile >> userId >> movieId >> date >> rating) {
          double r = compute_rating(userId, movieId, user_info, top_q, num_neighbors);
          outfile << userId << " " << movieId << " " << date << " " << r << endl;
      }

  }

   // delete[] all_data;
    delete[] user_info;
    infile.close();
    outfile.close();
    return 0;
}
