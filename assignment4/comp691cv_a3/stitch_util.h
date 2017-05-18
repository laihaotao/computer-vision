#pragma once
#include<opencv2/core.hpp>
#include"util.h"
using namespace cv;

void project (float in_x, float in_y, Mat const& H, float& out_x, float& out_y);
//Does not handle aliasing at all
void naiveWarpAffine (Mat const& input, Mat& output, Mat const& H);

void computeInlierCount(Mat const& H, vector<match> const& matches, int& numMatches, float inlierThreshold);
void RANSAC (vector<match> const& matches, int& numMatches, int numIterations, float inlierThreshold, Mat& hom, Mat& homInv, Mat& image1Display, Mat& image2Display, vector<Vec2f>& best_src, vector<Vec2f>& best_dst);

void findInliers (Mat const& H, vector<match> const& matches, float inlierThreshold, vector<Vec2f>& src, vector<Vec2f>& dst);

//The actual stitching stuff?
void stitchedDimensions (Mat const& img1, Mat const& img2, Mat const& homInv, int& min_x, int& min_y, int& max_x, int& max_y);
void stitchedAllocate (Mat const& img1, Mat const& img2, Mat const& homInv, Mat& stitched, int& start_x, int& start_y);
void stitch (Mat const& image1, Mat const& image2, Mat const& hom, Mat const& homInv, Mat& stitchedImage);