#include<iostream>
#include<cmath>
#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

/// find the middle point in both X and Y
Vec2d findCenter(Mat &input) {
    int c_max = 0, c_min = 99999, r_max = 0, r_min = 999999;
    for (int r = 0; r < input.rows; r++) {
        for (int c = 0; c < input.cols; c++) {
            if (input.at<uchar>(r, c) > (uchar) (255 * 0.9)) {
                if (r > r_max) {
                    r_max = r;
                }
                if (r < r_min) {
                    r_min = r;
                }
                if (c > c_max) {
                    c_max = c;
                }
                if (c < c_min) {
                    c_min = c;
                }
            }
        }
    }
    double x = c_min + (double) (c_max - c_min) / 2;
    double y = r_min + (double) (r_max - r_min) / 2;
    Vec2d center = Vec2d(x, y);
    return center;
}

double findChromeRadius(Mat &chrome) {
    int c_max = 0, c_min = 99999;
    for (int r = 0; r < chrome.rows; r++) {
        for (int c = 0; c < chrome.cols; c++) {
            if (chrome.at<uchar>(r, c) > 254) {
                if (c > c_max) {
                    c_max = c;
                }
                if (c < c_min) {
                    c_min = c;
                }
            }
        }
    }
    double r = (double) (c_max - c_min) / 2;
    return r;
}

/// subtract highlight center to chrome center
Vec3d findNormalChrome(Mat &chrome, Vec2d center_chrome, Vec2d center_highlight) {
    double r_chrome = findChromeRadius(chrome);

    double x = (center_highlight[0] - center_chrome[0]) / r_chrome;
    double y = (center_highlight[1] - center_chrome[1]) / r_chrome;
    double z = sqrt(1.0 - x * x - y * y);

    Vec3d N = Vec3d(x, y, z);
    return N;
}

Vec3d findLightingDir(Mat &mask_gray, Mat &highlight_gray) {
    Vec2d center_chrome = findCenter(mask_gray);
    Vec2d center_highlight = findCenter(highlight_gray);

    Vec3d N = findNormalChrome(mask_gray, center_chrome, center_highlight);

    Vec3d R = Vec3d(0, 0, 1);
    Vec3d L = 2 * (N.dot(R)) * N - R;

    return L;
}

vector<Vec2d> findMask(Mat &mask_gray) {
    vector<Vec2d> mask;
    for (int r = 0; r < mask_gray.rows; r++) {
        for (int c = 0; c < mask_gray.cols; c++) {
            if (mask_gray.at<uchar>(r, c) > (uchar) (255 * 0.9)) {
                mask.push_back(Vec2d(r, c));
            }
        }
    }
    return mask;
}

int main() {
    /// compute lighting direction
    Mat mask_chrome = imread("psmImages/chrome/chrome.mask.png", CV_LOAD_IMAGE_COLOR);
    Mat mask_chrome_gray;
    cvtColor(mask_chrome, mask_chrome_gray, CV_BGR2GRAY);
    Mat lights(3, 12, CV_64F);
    for (int i = 0; i < 12; i++) {
        string filename = "psmImages/chrome/chrome." + to_string(i) + ".png";
        Mat highlight = imread(filename, CV_LOAD_IMAGE_COLOR);
        Mat highlight_gray;
        cvtColor(highlight, highlight_gray, CV_BGR2GRAY);
        Vec3d L = findLightingDir(mask_chrome_gray, highlight_gray);
        lights.at<double>(0, i) = L[0];
        lights.at<double>(1, i) = L[1];
        lights.at<double>(2, i) = L[2];
    }


    /// compute normals
    string img_set = "cat";
    Mat img_mask = imread("psmImages/" + img_set + "/" + img_set + ".mask.png", CV_LOAD_IMAGE_COLOR);
    Mat img_mask_gray;
    cvtColor(img_mask, img_mask_gray, CV_BGR2GRAY);
    vector<Vec2d> mask_img = findMask(img_mask_gray);

    vector<Mat> pixels_all_set;
    for (int i = 0; i < 12; i++) {
        Mat img = imread("psmImages/" + img_set + "/" + img_set + "." + to_string(i) + ".png", CV_LOAD_IMAGE_COLOR);
        Mat tmp;
        cvtColor(img, tmp, CV_BGR2GRAY);
        pixels_all_set.push_back(tmp);
    }

    Mat normals(Size(img_mask_gray.cols, img_mask_gray.rows), CV_8UC3);
    normals = Vec3b(0, 0, 0);
    Mat normals_bgr[3];
    split(normals, normals_bgr);

    normals_bgr[0] = 0;
    normals_bgr[1] = 0;
    normals_bgr[2] = 255;

    vector<Mat> normals_vec;

    Mat lights12by3, I12by1;
    transpose(lights, lights12by3);
    for (int i = 0; i < mask_img.size(); i++) {  /// using least square method here
        Mat I(1, 12, CV_64F), L_inv, I_inv, IL(12, 3, CV_64F), IL_inv;
        for (int j = 0; j < 12; j++) {
            I.at<double>(0, j) = pixels_all_set[j].at<uchar>(mask_img[i][0], mask_img[i][1]) / 255.0;
        }
        transpose(I, I12by1);
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 3; j++) {
                IL.at<double>(i, j) = I.at<double>(0, i) * lights12by3.at<double>(i, j);
            }
        }
        invert(IL, IL_inv, DECOMP_SVD);
        Mat G = IL_inv * I12by1.mul(I12by1);
        double kd = sqrt(G.dot(G));
        Mat N = G / kd;

        normals_vec.push_back(N);
        double normals_b = -1 * (N.at<double>(0, 0) - (-1)) / 2.0 * 255; // x
        double normals_g = -1 * (N.at<double>(0, 1) - (-1)) / 2.0 * 255; // y
        double normals_r = N.at<double>(0, 2) * 255; //z
        normals_bgr[0].at<uchar>(mask_img[i][0], mask_img[i][1]) = (uchar) normals_b;
        normals_bgr[1].at<uchar>(mask_img[i][0], mask_img[i][1]) = (uchar) normals_g;
        normals_bgr[2].at<uchar>(mask_img[i][0], mask_img[i][1]) = (uchar) normals_r;
    }
    merge(normals_bgr, 3, normals);
    imwrite(img_set + "_normals.png", normals);
    imshow(img_set + "_normals", normals);
    waitKey(0);


    /// color albedo
    Mat albedos(Size(img_mask_gray.cols, img_mask_gray.rows), CV_8UC3);
    Mat albedos_bgr[3];
    split(albedos, albedos_bgr);
    albedos_bgr[0] = 0;
    albedos_bgr[1] = 0;
    albedos_bgr[2] = 0;

    vector<vector<Mat>> img_channels(3);
    for (int i = 0; i < 12; i++) {
        Mat img = imread("psmImages/" + img_set + "/" + img_set + "." + to_string(i) + ".png", CV_LOAD_IMAGE_COLOR);
        Mat bgr[3];
        split(img, bgr);
        img_channels[0].push_back(bgr[0]);
        img_channels[1].push_back(bgr[1]);
        img_channels[2].push_back(bgr[2]);
    }

    double min = 99999.9, max = -99999.9;
    vector<vector<double>> albedo_vec(3);
    for (int channel = 0; channel < 3; channel++) {
        for (int i = 0; i < mask_img.size(); i++) {
            Vec3d n = Vec3d(normals_vec[i].at<double>(0, 0), normals_vec[i].at<double>(0, 1),
                            normals_vec[i].at<double>(0, 2));
            Vec3d sum_up;
            double sum_down = 0.0;
            for (int j = 0; j < 12; j++) {
                double Ii = img_channels[channel][j].at<uchar>(mask_img[i][0], mask_img[i][1]) / 255.0;
                Vec3d Li = Vec3d(lights.at<double>(0, j), lights.at<double>(1, j), lights.at<double>(2, j));
                double Ji = n.dot(Li); /// using the calculated normals and lightings
                sum_up += Li * Ii;
                sum_down += Ji * Ji;
            }
            double albedo = sum_up.dot(n) / sum_down;
            albedo_vec[channel].push_back(albedo);

            /// keep track of the min and max of albedo for normalization
            if (albedo < min) {
                min = albedo;
            }
            if (albedo > max) {
                max = albedo;
            }
        }
    }

    /// normalization for albedos
    for (int channel = 0; channel < 3; channel++) {
        for (int i = 0; i < mask_img.size(); i++) {
            uchar pixel = (uchar) ((albedo_vec[channel][i] - min) / (max - min) * 255.0);
            albedos_bgr[channel].at<uchar>(mask_img[i][0], mask_img[i][1]) = pixel;
        }
    }

    merge(albedos_bgr, 3, albedos);
    imwrite(img_set + "_albedos.png", albedos);
    imshow(img_set + "_albedos", albedos);
    waitKey(0);

    /// depth
    Mat dx(normals.rows, normals.cols, CV_64F);
    Mat dy(normals.rows, normals.cols, CV_64F);
    /// solving the linear equations, and get delta x and delta y
    for (int r = 0; r < normals.rows; r++) {
        for (int c = 0; c < normals.cols; c++) {
            if (normals_bgr[2].at<uchar>(r, c) == 0) {
                normals_bgr[2].at<uchar>(r, c) = 1;
            }
            if (r == 0 && c < normals.cols - 1) {
                dx.at<double>(r, c) = (double) (-normals_bgr[0].at<uchar>(r, c) / normals_bgr[2].at<uchar>(r, c));
                dy.at<double>(r, c) = 0.0;
            } else if (r > 0 && c == normals.cols - 1) {
                dx.at<double>(r, c) = 0.0;
                dy.at<double>(r, c) = (double) (-normals_bgr[1].at<uchar>(r, c) / normals_bgr[2].at<uchar>(r, c));
            } else {
                dx.at<double>(r, c) = (double) (-normals_bgr[0].at<uchar>(r, c) / normals_bgr[2].at<uchar>(r, c));
                dy.at<double>(r, c) = (double) (-normals_bgr[1].at<uchar>(r, c) / normals_bgr[2].at<uchar>(r, c));
            }
        }
    }

    /// getting the depths in x and y through intergral
    Mat depthX(normals.rows, normals.cols, CV_64F);
    Mat depthY(normals.rows, normals.cols, CV_64F);
    double minX_depth = 99999.9, maxX_depth = -99999.9;
    double minY_depth = 99999.9, maxY_depth = -99999.9;
    for (int r = 0; r < normals.rows; r++) {
        for (int c = 0; c < normals.cols; c++) {
            if (c == 0) {
                depthX.at<double>(r, c) = 0;
            } else {
                depthX.at<double>(r, c) = depthX.at<double>(r, c - 1) + dx.at<double>(r, c);
            }
            if (depthX.at<double>(r, c) > maxX_depth) {
                maxX_depth = depthX.at<double>(r, c);
            }
            if (depthX.at<double>(r, c) < minX_depth) {
                minX_depth = depthX.at<double>(r, c);
            }
        }
    }
    for (int r = 0; r < normals.rows; r++) {
        for (int c = 0; c < normals.cols; c++) {
            if (r == 0) {
                depthY.at<double>(r, c) = 0;
            } else {
                depthY.at<double>(r, c) = depthY.at<double>(r - 1, c) + dy.at<double>(r, c);
            }
            if (depthY.at<double>(r, c) > maxY_depth) {
                maxY_depth = depthY.at<double>(r, c);
            }
            if (depthY.at<double>(r, c) < minY_depth) {
                minY_depth = depthY.at<double>(r, c);
            }
        }
    }

    /// normalize the depths
    Mat depth_X(normals.rows, normals.cols, CV_8U);
    Mat depth_Y(normals.rows, normals.cols, CV_8U);
    for (int r = 0; r < normals.rows; r++) {
        for (int c = 0; c < normals.cols; c++) {
            depth_X.at<uchar>(r, c) = (uchar) (255 -
                                               ((depthX.at<double>(r, c) - minX_depth) / (maxX_depth - minX_depth) *
                                                255.0));
            depth_Y.at<uchar>(r, c) = (uchar) (255 -
                                               ((depthY.at<double>(r, c) - minY_depth) / (maxY_depth - minY_depth) *
                                                255.0));
        }
    }

    Mat depth_X_weighted = (depth_X + 256) / 2;
    Mat depth_Y_weighted = (depth_Y + 256) / 2;
    Mat depth = (depth_X_weighted + depth_Y_weighted) / 2;
    imshow(img_set + "_depth_X (without albedo)", depth_X_weighted);
    imwrite(img_set + "_depth_X_without_albedo.png", depth_X_weighted);
    waitKey(0);
    imshow(img_set + "_depth_Y (without albedo)", depth_Y_weighted);
    imwrite(img_set + "_depth_Y_without_albedo.png", depth_Y_weighted);
    waitKey(0);
    imshow(img_set + "_depth (without albedo)", depth);
    imwrite(img_set + "_depth_without_albedo.png", depth);
    waitKey(0);

    Mat depth_X_albedo_bgr[3], depth_Y_albedo_bgr[3], depth_X_albedo, depth_Y_albedo, depth_albedo;
    for (int channel = 0; channel < 3; channel++) {
        depth_X_albedo_bgr[channel] = (depth_X_weighted / 255.0).mul(albedos_bgr[channel]);
        depth_Y_albedo_bgr[channel] = (depth_Y_weighted / 255.0).mul(albedos_bgr[channel]);
    }
    merge(depth_X_albedo_bgr, 3, depth_X_albedo);
    merge(depth_Y_albedo_bgr, 3, depth_Y_albedo);
    depth_albedo = (depth_X_albedo + depth_Y_albedo) / 2;
    imshow(img_set + "_depth_X (with albedo)", depth_X_albedo);
    imwrite(img_set + "_depth_X_with_albedo.png", depth_X_albedo);
    waitKey(0);
    imshow(img_set + "_depth_Y (with albedo)", depth_Y_albedo);
    imwrite(img_set + "_depth_Y_with_albedo.png", depth_Y_albedo);
    waitKey(0);
    imshow(img_set + "_depth (with albedo)", depth_albedo);
    imwrite(img_set + "_depth_with_albedo.png", depth_albedo);
    waitKey(0);

    return 0;
}
