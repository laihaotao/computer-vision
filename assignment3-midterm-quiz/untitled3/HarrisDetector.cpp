#include"HarrisDetector.h"
#include<iostream>

using namespace std;
using namespace cv;

HarrisDetector::HarrisDetector(
        char const *filename,
        float sigma,
        float min_response_threshold, float max_response_threshold, int local_maximum_half_size,
        float admissable_response_ratio
) :
        filename(filename),
        sigma(sigma),
        min_response_threshold(min_response_threshold),
        max_response_threshold(max_response_threshold),
        local_maximum_half_size(local_maximum_half_size),
        admissable_response_ratio(admissable_response_ratio) {}

bool HarrisDetector::find_features() {
    //Load image
    src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find image";
        return false;
    }
    //Convert to gray-scale
    cvtColor(src, gray, CV_BGR2GRAY);

    //Convert to floating point with range [0.0f, 255.0f]
    gray.convertTo(gray_f, CV_32F);
    float f = gray_f.at<float>(0, 0);
    //Derivative in both axes with range [-255.0f, 255.0f]
    dx = derivative_x(gray_f);
    dy = derivative_y(gray_f);

    //Element-wise multiplication, set up for harris matrix, which has the form:
    //dxx dxy
    //dxy dyy
    dxx = dx.mul(dx); //[0.0f, 255^2] because we're effectively squaring each element
    dyy = dy.mul(dy); //[0.0f, 255^2] because we're effectively squaring each element
    dxy = dx.mul(dy); //[-255^2, 255^2]

    //3 multiplier is arbitrary, will smooth our derivative images
    gaussian = gaussianKernelNormalized((int) (sigma * 3), sigma);

    //Will be used for keypoint histogram computation
    //We want to use dx and dy and not dxx and dyy because
    //dx and dy preserve the direction of the gradient.
    //dxx and dyy only point in the 1st quadrant.
    filter2D(dx, dx_smoothed, -1, gaussian); //[-255.0f, 255.0f]
    filter2D(dy, dy_smoothed, -1, gaussian); //[-255.0f, 255.0f]

    //Will be used for harris matrix.
    filter2D(dxx, dxx_smoothed, -1, gaussian); //[0.0f, 255^2]
    filter2D(dyy, dyy_smoothed, -1, gaussian); //[0.0f, 255^2]
    filter2D(dxy, dxy_smoothed, -1, gaussian); //[-255^2, 255^2]

    find_responses();
    find_local_maximum_responses();
    bool b = find_corners_and_features();
    return b;
}

void HarrisDetector::find_responses() {
    //0.0f U [min_response_threshold, max_response_threshold]
    responses = Mat(src.rows, src.cols, CV_32F, 0.0f);
    max_response = 0.0f;
    min_response = 999999999999.0f;
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            //2d matrix, H
            //dxx dxy
            //dxy dyy
            float a = dxx_smoothed.at<float>(row, col);
            float b = dxy_smoothed.at<float>(row, col);
            float c = b;
            float d = dyy_smoothed.at<float>(row, col);

            float e1;
            float e2;
            eigen2x2(a, b, c, d, e1, e2);
            float response = (e1 * e2) / (e1 + e2); // det(H)/trace(H)
            if (response > max_response) {
                max_response = response;
            }
            if (response < min_response) {
                min_response = response;
            }
            if (isnan(response) || response < min_response_threshold || response > max_response_threshold) {
                continue;
            }
            responses.at<float>(row, col) = response;
        }
    }
    cout << "responses found" << endl;
    cout << "min response: " << min_response << endl;
    cout << "max response: " << max_response << endl;
}

void HarrisDetector::find_local_maximum_responses() {
    //0.0f U [min_response_threshold, max_response_threshold]
    local_maximum_responses = Mat(src.rows, src.cols, CV_32F, 0.0f);
    max_response = 0.0f;
    min_response = 999999999999.0f;
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            float response = responses.at<float>(row, col);
            if (response < min_response_threshold) {
                continue;
            }

            if (is_local_maximum(responses, local_maximum_half_size, row, col)) {
                if (response > max_response) {
                    max_response = response;
                }
                if (response < min_response) {
                    min_response = response;
                }
                local_maximum_responses.at<float>(row, col) = response;
                local_maximum_point_arr.push_back({row, col, response});
            }
        }
    }
    cout << "locally maximal responses found" << endl;
    cout << "min response: " << min_response << endl;
    cout << "max response: " << max_response << endl;
    sort(local_maximum_point_arr.begin(), local_maximum_point_arr.end(), [](auto a, auto b) {
        return a.response > b.response;
    });
}

bool HarrisDetector::find_corners_and_features() {
    if (local_maximum_point_arr.size() == 0) {
        return false;
    }
    /*//iqr = interquartile range
    //We're finding an upper fence and
    //only admitting points that are within a certain ratio of the upper fence
    int q1_index = local_maximum_point_arr.size() * 3 / 4;
    int q3_index = local_maximum_point_arr.size() / 4;
    float q1_estimate = local_maximum_point_arr[q1_index].response;
    float q3_estimate = local_maximum_point_arr[q3_index].response;
    float iqr = q3_estimate - q1_estimate;
    float upper = q3_estimate + iqr * 3;

    int start_at = 0;
    for (int i = 0; i < local_maximum_point_arr.size(); ++i) {
        float diff = local_maximum_point_arr[i].response - upper;
        if (diff <= 0.0f) {
            break;
        }
        start_at = i + 1;
    }*/
    //0 U 255
    corners = Mat::zeros(src.rows, src.cols, CV_8UC1);
    //[0, 255]
    corners2 = src.clone();
    //float const admissable_response = local_maximum_point_arr[start_at].response * admissable_response_ratio;*/
    Mat gaussian = gaussianKernelNormalized(8, 1.5f * 16, false);
    for (int i = 0; i < local_maximum_point_arr.size(); ++i) {
        auto rp = local_maximum_point_arr[i];
        //if (rp.response > admissable_response) {
        corners.at<uchar>(rp.r, rp.c) = 255;
        circle(corners2, Point(rp.c, rp.r), 2, Scalar(255, 0, 0), 3);

        vector<float> rad_arr;
        dominant_orientations(dx_smoothed, dy_smoothed, rp.r, rp.c, sigma, rad_arr);
        for (int i = 0; i < rad_arr.size(); ++i) {
            features.push_back({ // descriptor
                                       rp,
                                       histogram128(
                                               dx_smoothed, dy_smoothed, gaussian,
                                               rp.r, rp.c,
                                               rad_arr[i]
                                       )
                               });
        }
        //}
    }
    return true;
}