#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/features2d.hpp>
#include"util.h"
#include"HarrisDetector.h"
#include"stitch_util.h"

using namespace std;
using namespace cv;

// put all display stuff in this function
void display_process(HarrisDetector const &d) {
    print(d.min_response);
    print(d.max_response);
    show(d.src);
    show(d.gray);
    //show(d.gray_f);
    d.dx += 255;
    d.dx /= 255 * 2;
    show(d.dx);

    d.dy += 255;
    d.dy /= 255 * 2;
    show(d.dy);
    //show(d.dxx);
    //show(d.dyy);
    //show(d.dxy);
    //show(d.gaussian);
    d.dx_smoothed += 255;
    d.dx_smoothed /= 255 * 2;
    show(d.dx_smoothed);

    d.dy_smoothed += 255;
    d.dy_smoothed /= 255 * 2;
    show(d.dy_smoothed);

    d.dxx_smoothed /= 255 * 255;
    show(d.dxx_smoothed);

    d.dyy_smoothed /= 255 * 255;
    show(d.dyy_smoothed);

    d.dxy_smoothed += 255 * 255;
    d.dxy_smoothed /= 255 * 255 * 2;
    show(d.dxy_smoothed);

    d.responses_before_threshold /= d.max_response;
    Mat test;
    d.responses_before_threshold.convertTo(test, CV_8UC3, 255);
    imwrite("test.png", test);
    show(d.responses_before_threshold);

    d.responses /= d.max_response;
    show(d.responses);

    d.local_maximum_responses /= d.max_response;
    show(d.local_maximum_responses);

    show(d.corners);
    show(d.corners2);
    /*
    print(d.min_response);
    print(d.max_response);
    show(d.src);
    show(d.gray);
    show(d.gray_f);
    show(d.dx);
    show(d.dy);
    show(d.dxx);
    show(d.dyy);
    show(d.dxy);
    show(d.gaussian);
    show(d.dx_smoothed);
    show(d.dy_smoothed);
    show(d.dxx_smoothed);
    show(d.dyy_smoothed);
    show(d.dxy_smoothed);
    show(d.responses);
    show(d.local_maximum_responses);
    show(d.corners);
    show(d.corners2);//*/
}

void destroy_process() {
    unshow(d.src);
    unshow(d.gray);
    unshow(d.gray_f);
    unshow(d.dx);
    unshow(d.dy);
    unshow(d.dxx);
    unshow(d.dyy);
    unshow(d.dxy);
    unshow(d.gaussian);
    unshow(d.dx_smoothed);
    unshow(d.dy_smoothed);
    unshow(d.dxx_smoothed);
    unshow(d.dyy_smoothed);
    unshow(d.dxy_smoothed);
    unshow(d.responses);
    unshow(d.local_maximum_responses);
    unshow(d.corners);
    unshow(d.corners2);
}

// pics in the image set, including the optimized parameter

/*
const char *filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/comp691cv_a3/assignment3_sample_images/Boxes.png";
const char *filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/comp691cv_a3/assignment3_sample_images/Boxes copy.png";
int window_sigma = 2;
int window_min_response = 9;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 5;//*/

//*
const char *filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/comp691cv_a3/assignment3_sample_images/Rainier1.png";
const char *filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/comp691cv_a3/assignment3_sample_images/Rainier2.png";
int window_sigma = 2;
int window_min_response = 9;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 5;
int window_numIterations = 2000;//100;
int window_inlierThreshold = 10000;//*/

/*
const char *filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/yosemite/Yosemite1.jpg";
const char *filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/yosemite/Yosemite2.jpg";
int window_sigma = 2;
int window_min_response = 9;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 5;//*/

/*
const char *filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/assignment3_sample_images/Hanging1.png";
const char *filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment4/assignment3_sample_images/Hanging3.png";
int window_sigma = 2;
int window_min_response = 9;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 5;//*/


/*
const char* filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3-midterm-quiz/untitled3/graf/img1.ppm";
const char* filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3-midterm-quiz/untitled3/graf/img2.ppm";
int window_sigma = 3;
int window_min_response = 12;
int window_max_response = 100;
int window_local_maximum_half_size = 6;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 12;
int window_numIterations = 100;
int window_inlierThreshold = 10000;//*/

/*
const char* filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/panorama/pano1_0008.png";
const char* filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/panorama/pano1_0009.png";
int window_sigma = 2;
int window_min_response = 6;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 4;//*/

/*
const char *filename_a = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/square_00.png";
const char *filename_b = "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment3/untitled3/square_01.png";
int window_sigma = 2;
int window_min_response = 10;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 20;//*/

/*
int window_sigma = 5;
int window_min_response = 10;
int window_max_response = 100;
int window_local_maximum_half_size = 1;
int window_admissable_response_ratio = 0;
int window_max_distance_threshold = 10;*/

int window_detailed = 0;
int window_filter_matches = 1;
int window_find_matches = 1;
HarrisDetector* prv_detector_a = nullptr;
HarrisDetector* prv_detector_b = nullptr;
vector<match>* prv_matches = nullptr;
bool calculate_lock = false; // something like a mutex lock

int main() {
    void (*on_trackbar)(int) = [](int) {
        if (calculate_lock) {
            cout << "Tweak after current computation ends" << endl;
            return;
        }
        calculate_lock = true;

        if (window_min_response < 1) {
            window_min_response = 1;
        }
        if (window_local_maximum_half_size < 1) {
            window_local_maximum_half_size = 1;
        }
        float sigma = window_sigma;
        float min_response_threshold = (float) window_min_response * 1000;
        float max_response_threshold = (float) window_max_response * 1000;
        int local_maximum_half_size = window_local_maximum_half_size;
        float admissable_response_ratio = (float) window_admissable_response_ratio / 100;
        float max_distance_threshold = window_max_distance_threshold / 100.0f;
        bool detailed = (window_detailed == 1);
        bool do_filter_matches = (window_filter_matches == 1);
        int numIterations = window_numIterations*5;
        float inlierThreshold = window_inlierThreshold / 1000.0f; // 1.0f;

        bool find_matches = (window_find_matches == 1);

        if (find_matches) {
            cout << "sigma = " << sigma << endl;
            cout << "min_response_threshold = " << min_response_threshold << endl;
            cout << "max_response_threshold = " << max_response_threshold << endl;
            cout << "local_maximum_half_size = " << local_maximum_half_size << endl;
            cout << "admissable_response_ratio = " << admissable_response_ratio << endl;
            cout << "max_distance_threshold = " << max_distance_threshold << endl;
            cout << "numIterations = " << numIterations << endl;
            cout << "inlierThreshold = " << inlierThreshold << endl;

            HarrisDetector& detector_a = *new HarrisDetector(filename_a, sigma, min_response_threshold, max_response_threshold,
                                      local_maximum_half_size, admissable_response_ratio);
            HarrisDetector& detector_b = *new HarrisDetector(filename_b, sigma, min_response_threshold, max_response_threshold,
                                      local_maximum_half_size, admissable_response_ratio);//*/
            if (!detector_a.find_features()) {
                cvDestroyWindow("detector_a.corners2");
                cvDestroyWindow("detector_b.corners2");
                cvDestroyWindow("match_mat_a");
                cvDestroyWindow("match_mat_b");
                calculate_lock = false;
                return;
            }
            if (detailed) {
                display_process(detector_a);
                destroy_process();
            }

            if (!detector_b.find_features()) {
                cvDestroyWindow("detector_a.corners2");
                cvDestroyWindow("detector_b.corners2");
                cvDestroyWindow("match_mat_a");
                cvDestroyWindow("match_mat_b");
                calculate_lock = false;
                return;
            }
            if (detailed) {
                display_process(detector_b);
                destroy_process();
            }

            vector<match>& matches = *new vector<match>();
            match_features(detector_a.features, detector_b.features, max_distance_threshold, do_filter_matches,
                           matches);

            Mat match_mat_a = detector_a.corners2.clone();
            Mat match_mat_b = detector_b.corners2.clone();
            for (int i = 0; i < matches.size(); ++i) {
                match m = matches[i];
                feature a = detector_a.features[m.index_a];
                feature b = detector_b.features[m.index_b];
                line(match_mat_a, CvPoint(a.p.c, a.p.r), CvPoint(b.p.c, b.p.r), Scalar(0));
                line(match_mat_b, CvPoint(a.p.c, a.p.r), CvPoint(b.p.c, b.p.r), Scalar(0));

                circle(match_mat_a, CvPoint(a.p.c, a.p.r), 3, Scalar(0));
                circle(match_mat_b, CvPoint(a.p.c, a.p.r), 3, Scalar(0));
            }
            show_nowait(detector_a.corners2);
            show_nowait(detector_b.corners2);
            show_nowait(match_mat_a);
            show_nowait(match_mat_b);

            if (prv_detector_a) {
                delete prv_detector_a;
            }
            if (prv_detector_b) {
                delete prv_detector_b;
            }
            if (prv_matches) {
                delete prv_matches;
            }
            prv_detector_a = &detector_a;
            prv_detector_b = &detector_b;
            prv_matches = &matches;

            //vector<Vec2f> inlier_src;
            //vector<Vec2f> inlier_dst;
            //findInliers(hom, matches, inlierThreshold, inlier_src, inlier_dst);

            // using drawMatches to visualize the matches
            vector<KeyPoint> k_a;
            vector<KeyPoint> k_b;
            vector<DMatch> m;
            Mat mat_matches;
            /*
            for (int i=0; i<inlier_src.size(); ++i) {
                k_a.push_back(KeyPoint(inlier_src[i][0], inlier_src[i][1], 1.0f));
                k_b.push_back(KeyPoint(inlier_dst[i][0], inlier_dst[i][1], 1.0f));
                m.push_back(DMatch(i, i, 1));
            }*/

            //cout << "Inlier Count: " << inlier_src.size() << endl;

            for (int i = 0; i < matches.size(); ++i) {
                response_point p = detector_a.features[matches[i].index_a].p;
                k_a.push_back(KeyPoint(p.c, p.r, 1.0f));

                response_point p2 = detector_b.features[matches[i].index_b].p;
                k_b.push_back(KeyPoint(p2.c, p2.r, 1.0f));

                m.push_back(DMatch(i, i, 1));
            }

            drawMatches(detector_a.src, k_a, detector_b.src, k_b, m, mat_matches);
            show_nowait(mat_matches);
        }
        if (prv_detector_a && prv_detector_b && prv_matches) {
            HarrisDetector& detector_a = *prv_detector_a;
            HarrisDetector& detector_b = *prv_detector_b;
            vector<match>& matches = *prv_matches;
            int numMatches;
            Mat hom;
            Mat homInv;
            Mat image1;
            Mat image2;
            vector<Vec2f> best_src;
            vector<Vec2f> best_dst;
            RANSAC(matches, numMatches, numIterations, inlierThreshold, hom, homInv, image1, image2, best_src, best_dst);

            vector<KeyPoint> k_a;
            vector<KeyPoint> k_b;
            vector<DMatch> m;
            Mat mat_best;

            for (int i=0; i<best_src.size(); ++i) {
                k_a.push_back(KeyPoint(best_src[i][0], best_src[i][1], 1.0f));
                k_b.push_back(KeyPoint(best_dst[i][0], best_dst[i][1], 1.0f));
                m.push_back(DMatch(i, i, 1));
            }
            drawMatches(detector_a.src, k_a, detector_b.src, k_b, m, mat_best);
            show_nowait(mat_best);

            Mat stitched;
            stitch(detector_a.src, detector_b.src, hom, homInv, stitched);
            if (stitched.rows > 0 && stitched.cols > 0) {
                show_nowait(stitched);
            }
        }
        calculate_lock = false;
        cout << "computation done" << endl;
    };

    // tuning UI
    namedWindow("tweak");
    resizeWindow("tweak", 480, 400);
    cvCreateTrackbar("sigma", "tweak", &window_sigma, 20, on_trackbar);
    cvCreateTrackbar("min response", "tweak", &window_min_response, 20, on_trackbar);
    cvCreateTrackbar("max response", "tweak", &window_max_response, 100, on_trackbar);
    cvCreateTrackbar("local max window", "tweak", &window_local_maximum_half_size, 20, on_trackbar);
    cvCreateTrackbar("admissable ratio", "tweak", &window_admissable_response_ratio, 100, on_trackbar);
    cvCreateTrackbar("max distance threshold", "tweak", &window_max_distance_threshold, 50, on_trackbar);
    cvCreateTrackbar("detailed", "tweak", &window_detailed, 1, on_trackbar);
    cvCreateTrackbar("filter matches", "tweak", &window_filter_matches, 1, on_trackbar);
    cvCreateTrackbar("numIterations", "tweak", &window_numIterations, 100, on_trackbar);
    cvCreateTrackbar("inlierThreshold", "tweak", &window_inlierThreshold, 10000, on_trackbar);
    cvCreateTrackbar("find matches", "tweak", &window_find_matches, 1, on_trackbar);

    on_trackbar(0);
    while (waitKey(0) != 27) {} //ESC key
    return 0;
}