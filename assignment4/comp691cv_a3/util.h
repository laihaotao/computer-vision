#pragma once
#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

struct response_point {
    int r;
    int c;
    float response;
};
struct feature {
    response_point p;
    Vec<float, 128> f;
};
struct match {
    int index_a;
    int index_b;
    float distance;
    int a_r;
    int a_c;
    int b_r;
    int b_c;
};
float const PI = 3.1415926535897932384626433832795028841971693993751f;

void show_window (char const* const name, Mat const& mat);

#define show(var) show_window((#var), var); waitKey(0)
#define show_nowait(var) show_window((#var), var)
#define unshow(var) destroyWindow(#var)
#define print(var) std::cout << (#var) << ": " << var << endl

//From: http://stackoverflow.com/questions/4633177/c-how-to-wrap-a-float-to-the-interval-pi-pi
//Hopefully wraps to [0.0f, divisor)
float fwrap (float x, float divisor);

Mat gaussianKernelNormalized (int half_size, float sigma, bool make_odd=true);
Mat derivative_x (Mat const& input);
Mat derivative_y (Mat const& input);
bool is_local_maximum (Mat const& input, int check_half_size, int row, int col);

//[-b +- sqrt(b^2-4ac)]/2a
void quadratic_roots (double a, double b, double c, float& r1, float& r2);
void eigen2x2 (float a, float b, float c, float d, float& e1, float& e2);
//location of top-left of 4x4 grid = r, c
//dx and dy must have same rows and cols, have 1 float channel
Vec<float, 8> histogram4x4 (Mat const& dx_mat, Mat const& dy_mat, Mat const& gaussian, int gaussian_r, int gaussian_c, int r, int c, float const rad);
float magnitude (Vec<float, 128> const& v);
void normalize (Vec<float, 128>& v);
bool truncate_above (Vec<float, 128>& v, float threshold=0.2f);
bool hasnan (Vec<float, 128>& v);
bool is_zero (Vec<float, 128>& v);
Vec<float, 128> histogram128_from_4x4arr (Vec<float, 8> (*histogram_arr)[4]);
//keypoint location = r, c
Vec<float, 128> histogram128 (Mat const& dx, Mat const& dy, Mat const& gaussian, int r, int c, float rad);
//Returns a rad rotation, [-pi, pi)
void dominant_orientations (
    Mat const& dx_mat, Mat const& dy_mat,
    int r, int c,
    float sigma, vector<float>& rad_results
);

bool detect_features (char const* filename, vector<feature>& results, Mat& result_corners2);
bool filter_matches (vector<match>& intermediate, vector<match>& result, bool(*pred)(match, match));
void match_features (
    vector<feature> const& features_a, vector<feature> const& features_b,
    float max_distance_threshold, bool do_filter_matches,
    vector<match>& result
);