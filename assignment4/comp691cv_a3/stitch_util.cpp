#include"stitch_util.h"
#include<opencv2/calib3d.hpp>

//Mat must be type CV_32F, 1 channel, 3x3
void project (float in_x, float in_y, Mat const& H, float& out_x, float& out_y) {
    if (H.type() != CV_32F) {
        throw "Please make it CV_32F, kthxbye";
    }
    Vec3f in;
    in[0] = in_x;
    in[1] = in_y;
    in[2] = 1;
    Mat in_mat = Mat(in);

    Mat result = H * in_mat;

    out_x = result.at<float>(0, 0);
    out_y = result.at<float>(1, 0);
}
void naiveWarpAffine (Mat const& input, Mat& output, Mat const& H) {
    for (int r=0; r<input.rows; ++r) {
        for (int c=0; c<input.cols; ++c) {
            float x;
            float y;
            project(c, r, H, x, y);

            int out_r = y;
            int out_c = x;
            if (out_r < 0 || out_r >= output.rows || out_c < 0 || out_c >= output.cols) {
                continue;
            }
            output.at<unsigned char>(out_r, out_c) = input.at<unsigned char>(r, c);
        }
    }
}

void computeInlierCount(Mat const& H, vector<match> const& matches, int& numMatches, float inlierThreshold) {
    numMatches = 0;
    for (int i=0; i<matches.size(); ++i) {
        match m = matches[i];
        float out_r;
        float out_c;
        project(m.a_c, m.a_r, H, out_c, out_r);

        float dr = out_r-m.b_r;
        float dc = out_c-m.b_c;
        float distance = sqrt(dr*dr + dc*dc);

        if (distance < inlierThreshold) {
            ++numMatches;
        }
    }
}
void findInliers (Mat const& H, vector<match> const& matches, float inlierThreshold, vector<Vec2f>& src, vector<Vec2f>& dst) {
    for (int i=0; i<matches.size(); ++i) {
        match m = matches[i];
        float out_r;
        float out_c;
        project(m.a_c, m.a_r, H, out_c, out_r);

        // cout<<"project: "<<out_c<<", "<<out_r<<endl;

        float dr = out_r-m.b_r;
        float dc = out_c-m.b_c;
        float distance = sqrt(dr*dr + dc*dc);

        if (distance < inlierThreshold) {
            src.push_back(Vec2f(m.a_c, m.a_r));
            dst.push_back(Vec2f(m.b_c, m.b_r));
        }
    }
}
bool roughly_collinear (Vec2f a_src, Vec2f a_dst, Vec2f b_src, Vec2f b_dst, float threshold) {
    Vec2f a_delta = a_dst - a_src;
    Vec2f b_delta = b_dst - b_src;

    float const THRESHOLD_ROUGHLY_VERTICAL = 100.0f;
    float const THRESHOLD_ROUGHLY_HORIZONTAL = 0.05f;

    bool a_rougly_vertical = false;
    float a_slope = 0.0f;

    if (a_delta[0] == 0.0f) {
        //a is a vertical line
        a_rougly_vertical = true;
    } else {
        float a_slope = a_delta[1]/a_delta[0];
        if (abs(a_slope) > THRESHOLD_ROUGHLY_VERTICAL) {
            a_rougly_vertical = true;
        }
    }

    bool b_rougly_vertical = false;
    float b_slope = 0.0f;

    if (b_delta[0] == 0.0f) {
        //b is a vertical line
        b_rougly_vertical = true;
    } else {
        float b_slope = b_delta[1]/b_delta[0];
        if (abs(b_slope) > THRESHOLD_ROUGHLY_VERTICAL) {
            b_rougly_vertical = true;
        }
    }

    if (a_rougly_vertical && b_rougly_vertical) {
        return true;
    }

    if (abs(a_slope) < THRESHOLD_ROUGHLY_HORIZONTAL && abs(b_slope) < THRESHOLD_ROUGHLY_HORIZONTAL) {
        return true;
    }

    return abs(a_slope-b_slope) < threshold;
}
bool roughly_colliner (vector<Vec2f> const& src, vector<Vec2f> const& dst, Vec2f& first_src, Vec2f& first_dst, float threshold) {
    int count = 0;
    for (int a=0; a<src.size(); ++a) {
        for (int b=a+1; b<src.size(); ++b) {
            if (roughly_collinear(src[a], dst[a], src[b], dst[b], threshold)) {
                if (count == 0) {
                    first_src = src[a];
                    first_dst = dst[a];
                }
                ++count;
            }

        }
    }
    return count > 1; //could tweak this to >0 ?
}
void RANSAC (vector<match> const& matches, int& numMatches, int numIterations, float inlierThreshold, Mat& hom, Mat& homInv, Mat& image1Display, Mat& image2Display, vector<Vec2f>& best_src, vector<Vec2f>& best_dst) {
    int homography_src_count = 4;
    int rand_inc = matches.size()/homography_src_count;
    if (rand_inc == 0) {
        cout << "no matches..." << endl;
        hom = Mat::eye(3, 3, CV_32F);
        homInv = hom;
        return;
    }
    int best_inlier_count = 0;
    Mat best_h;
    bool best_is_collinear = false;

    for (int i=0; i<numIterations; ++i) {
        vector<Vec2f> src;
        vector<Vec2f> dst;

        // Randomly select 4 pairs of potentially matching points from "matches".
        {
            int index = rand() % rand_inc;
            for (int n=0; n<homography_src_count && index < matches.size(); ++n) {
                match m = matches[index];
                src.push_back(Vec2f(m.a_c, m.a_r));
                dst.push_back(Vec2f(m.b_c, m.b_r));
                index += (rand() % rand_inc) + 1;
            }
        }
        Mat h = findHomography(src, dst, 0); // Compute the homography relating the four selected matches
        h.convertTo(h, CV_32F);

        // Using the computed homography, compute the number of inliers
        int inlier_count;
        computeInlierCount(h, matches, inlier_count, inlierThreshold);
        // If this homography produces the highest number of inliers, store it as the best homography.
        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_h = h;
            best_is_collinear = false;
            //best_src = src;
            //best_dst = dst;
        }

        // almost the same as above
        Vec2f collinear_src;
        Vec2f collinear_dst;
        if (false){//roughly_colliner(src, dst, collinear_src, collinear_dst, 0.05)) {
            //Pretend we can just use translation.. ?

            Mat h = Mat::eye(3, 3, CV_32F);
            Vec2f to_dst = collinear_dst-collinear_src;
            h.at<float>(0, 2) = to_dst[0];
            h.at<float>(1, 2) = to_dst[1];

            h.convertTo(h, CV_32F);

            int inlier_count;
            computeInlierCount(h, matches, inlier_count, inlierThreshold);
            if (inlier_count > best_inlier_count) {
                best_inlier_count = inlier_count;
                best_h = h;
                best_is_collinear = true;
                //best_src = src;
                //best_dst = dst;
            }
        }

    }

    if (best_inlier_count == 0) {
        cout << "No inliers..." << endl;
        hom = Mat::eye(3, 3, CV_32F);
        homInv = Mat::eye(3, 3, CV_32F);
        return;
    }
    cout << "best_inlier_count: " << best_inlier_count << endl;
    cout << "best_hom" << endl;
    for (int r=0; r<3; ++r) {
        for (int c=0; c<3; ++c) {
            cout << best_h.at<float>(r, c) << ", ";
        }
        cout << endl;
    }

    // Given the highest scoring homography, once again find all the inliers.
    vector<Vec2f> inlier_src;
    vector<Vec2f> inlier_dst;
    findInliers(best_h, matches, inlierThreshold, inlier_src, inlier_dst);

    best_src = inlier_src;
    best_dst = inlier_dst;
    if (inlier_src.size() <= 1) {
        cout << "Too few inliers..." << endl;
        hom = best_h;
        homInv = hom.inv();
        return;
    }
    if (best_is_collinear) {
        hom = best_h;
    } else {
        // Compute a new refined homography using all of the inliers (not just using four points as you did previously)
        hom = findHomography(inlier_src, inlier_dst, 0);
        if (hom.rows == 0 || hom.cols == 0) {
            cout << "Failed to find homography..." << endl;
            hom = Mat::eye(3, 3, CV_32F);
            homInv = Mat::eye(3, 3, CV_32F);
            return;
        }
    }

    // Compute an inverse homography and print them
    homInv = hom.inv(DECOMP_SVD);
    hom.convertTo(hom, CV_32F);
    homInv.convertTo(homInv, CV_32F);
    cout << "hom" << endl;
    for (int r=0; r<3; ++r) {
        for (int c=0; c<3; ++c) {
            cout << hom.at<float>(r, c) << ", ";
        }
        cout << endl;
    }
    cout << "homInv" << endl;
    for (int r=0; r<3; ++r) {
        for (int c=0; c<3; ++c) {
            cout << homInv.at<float>(r, c) << ", ";
        }
        cout << endl;
    }
}

void stitchedDimensions (Mat const& img1, Mat const& img2, Mat const& homInv, int& min_x, int& min_y, int& max_x, int& max_y) {
    min_x = 0;
    min_y = 0;
    max_x = img1.cols;
    max_y = img1.rows;

    float out_x;
    float out_y;

    project(0, 0, homInv, out_x, out_y);
    min_x = min(min_x, (int)out_x);
    min_y = min(min_y, (int)out_y);
    max_x = max(max_x, (int)out_x);
    max_y = max(max_y, (int)out_y);

    project(img2.cols, 0, homInv, out_x, out_y);
    min_x = min(min_x, (int)out_x);
    min_y = min(min_y, (int)out_y);
    max_x = max(max_x, (int)out_x);
    max_y = max(max_y, (int)out_y);

    project(img2.cols, img2.rows, homInv, out_x, out_y);
    min_x = min(min_x, (int)out_x);
    min_y = min(min_y, (int)out_y);
    max_x = max(max_x, (int)out_x);
    max_y = max(max_y, (int)out_y);

    project(0, img2.rows, homInv, out_x, out_y);
    min_x = min(min_x, (int)out_x);
    min_y = min(min_y, (int)out_y);
    max_x = max(max_x, (int)out_x);
    max_y = max(max_y, (int)out_y);
}
void stitchedAllocate (Mat const& img1, Mat const& img2, Mat const& homInv, Mat& stitched, int& start_x, int& start_y) {
    int min_x;
    int min_y;
    int max_x;
    int max_y;
    stitchedDimensions(img1, img2, homInv, min_x, min_y, max_x, max_y);
    int w = max_x-min_x;
    int h = max_y-min_y;
    if (w <= 0 || h <= 0) {
        return;
    }
    if (w > img1.cols + img2.cols || h > img1.rows + img2.rows) {
        return;
    }
    stitched = Mat(h, w, CV_8UC3);
    stitched = 0;
    img1.copyTo(stitched(Rect(-min_x, -min_y, img1.cols, img1.rows)));
    start_x = min_x;
    start_y = min_y;
}
void stitch (Mat const& image1, Mat const& image2, Mat const& hom, Mat const& homInv, Mat& stitchedImage) {
    int start_x;
    int start_y;
    stitchedAllocate(image1, image2, homInv, stitchedImage, start_x, start_y);
    for (int r=0; r<stitchedImage.rows; ++r) {
        for (int c=0; c<stitchedImage.cols; ++c) {
            float out_x;
            float out_y;
            project(c+start_x, r+start_y, hom, out_x, out_y);
            int out_r = out_y;
            int out_c = out_x;
            if (out_r < 0 || out_r >= image2.rows || out_c < 0 || out_c >= image2.cols) {
                continue;
            }
            getRectSubPix(image2, Size(1, 1), Point2f(out_x, out_y), stitchedImage(Rect(c, r, 1, 1)));
        }
    }
}