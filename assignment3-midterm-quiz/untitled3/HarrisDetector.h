#pragma once
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include"util.h"
using namespace std;
using namespace cv;

class HarrisDetector {
    public:
    char const* filename;
    float const sigma;
    float const min_response_threshold;
    float const max_response_threshold;
    int const local_maximum_half_size;
    float const admissable_response_ratio;

    Mat src;
    Mat gray;
    Mat gray_f;
    Mat dx;
    Mat dy;
    Mat dxx;
    Mat dyy;
    Mat dxy;
    Mat gaussian;
    Mat dx_smoothed;
    Mat dy_smoothed;
    Mat dxx_smoothed;
    Mat dyy_smoothed;
    Mat dxy_smoothed;
    Mat responses;
    Mat local_maximum_responses;
    vector<response_point> local_maximum_point_arr;
    float max_response = 0.0f;
    float min_response = 999999999999.0f;
    int q1_index;
    int q3_index;
    float q1_est;
    float q3_est;
    float iqr;
    float upper;
    Mat corners;
    Mat corners2;
    vector<feature> features;
    //No scale invariance, therefore, sigma has to be explicitly tweaked and given
    //min_response_threshold should be > 0.0
    //local_maximum_half_size should be 1 or 2
    //admissable_response_ratio is just something to tweak. [0.0, 1.0], smaller value = more corners admitted
    HarrisDetector (
        char const* filename,
        float sigma,
        float min_response_threshold, float max_response_threshold, int local_maximum_half_size,
        float admissable_response_ratio
    );
    bool find_features ();

    private:
    void find_responses ();
    void find_local_maximum_responses ();
    bool find_corners_and_features();
};
