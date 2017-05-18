#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

void fillZero4BlueChannel(cv::Mat &blue);

void fillZero4GreenChannel(cv::Mat &green);

void fillZero4RedChannel(cv::Mat &red);

cv::Mat simpleLinearInterpolation4BR(cv::Mat &rb);

cv::Mat simpleLinearInterpolation4G(cv::Mat &green);

cv::Mat simpleDemosaic(cv::Mat &mosaic);

cv::Mat freemanImprvDemosaic(cv::Mat &mosaic);

int main() {
    cv::Mat mosaic;
    cv::Mat origin;
    mosaic = cv::imread("/Users/m.ding/Documents/concordia/courses/comp691cv/assignment1/comp691cv-a1/resources/oldwell_mosaic.png",
                        CV_LOAD_IMAGE_COLOR);
    origin = cv::imread("/Users/m.ding/Documents/concordia/courses/comp691cv/assignment1/comp691cv-a1/resources/oldwell.jpg", CV_LOAD_IMAGE_COLOR);
    if (!mosaic.data || !origin.data) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }

    // part 1: simple linear interpolation
    cv::Mat demosaic = simpleDemosaic(mosaic);
    cv::Mat diff = demosaic - origin + origin - demosaic;

    // part 2: improved approach
    cv::Mat demosaic2 = freemanImprvDemosaic(mosaic);
    cv::Mat diff2 = demosaic2 - origin + origin - demosaic2;

    // show the images
    cv::namedWindow("origin", cv::WINDOW_AUTOSIZE);
    cv::imshow("origin", origin);
    cv::waitKey(0);

    cv::namedWindow("demosaic", cv::WINDOW_AUTOSIZE);
    cv::imshow("demosaic", demosaic);
    cv::waitKey(0);

    cv::namedWindow("diff", cv::WINDOW_AUTOSIZE);
    cv::imshow("diff", diff);
    cv::waitKey(0);

    cv::namedWindow("demosaic2", cv::WINDOW_AUTOSIZE);
    cv::imshow("demosaic2", demosaic2);
    cv::waitKey(0);

    cv::namedWindow("diff2", cv::WINDOW_AUTOSIZE);
    cv::imshow("diff2", diff2);
    cv::waitKey(0);

    return 0;
}

cv::Mat simpleDemosaic(cv::Mat &mosaic) {
    cv::Mat bgr[3];         //destination array
    cv::split(mosaic, bgr); //split source

    // fill all ? to 0 for each channel respectively.
    fillZero4BlueChannel(bgr[0]);
    fillZero4GreenChannel(bgr[1]);
    fillZero4RedChannel(bgr[2]);

    cv::namedWindow("blue", cv::WINDOW_AUTOSIZE);
    cv::imshow("blue", bgr[0]);
    cv::waitKey(0);

    cv::namedWindow("green", cv::WINDOW_AUTOSIZE);
    cv::imshow("green", bgr[1]);
    cv::waitKey(0);

    cv::namedWindow("red", cv::WINDOW_AUTOSIZE);
    cv::imshow("red", bgr[2]);
    cv::waitKey(0);

    // use simple linear interpolation for each channel respectively
    cv::Mat demosaicBGR[3];
    demosaicBGR[0] = simpleLinearInterpolation4BR(bgr[0]);
    demosaicBGR[1] = simpleLinearInterpolation4G(bgr[1]);
    demosaicBGR[2] = simpleLinearInterpolation4BR(bgr[2]);

    // merge the 3 demosaic-ed channel
    cv::Mat demosaic;
    cv::merge(demosaicBGR, 3, demosaic);

    return demosaic;
}

cv::Mat freemanImprvDemosaic(cv::Mat &mosaic) {
    cv::Mat bgr[3];         //destination array
    cv::split(mosaic, bgr); //split source

    // fill all ? to 0 for each channel respectively.
    fillZero4BlueChannel(bgr[0]);
    fillZero4GreenChannel(bgr[1]);
    fillZero4RedChannel(bgr[2]);

    // use simple linear interpolation for each channel respectively
    cv::Mat demosaicBGR[3];
    demosaicBGR[0] = simpleLinearInterpolation4BR(bgr[0]);
    demosaicBGR[1] = simpleLinearInterpolation4G(bgr[1]);
    demosaicBGR[2] = simpleLinearInterpolation4BR(bgr[2]);

    // for part 2: an improvement of the simple linear interpolation approach proposed by Bill Freeman.
    cv::Mat rMinusG = demosaicBGR[2] - demosaicBGR[1];
    cv::Mat bMinusG = demosaicBGR[0] - demosaicBGR[1];

    cv::Mat mFilRMinusG, mFilBMinusG;
    cv::medianBlur(rMinusG, mFilRMinusG, 3);
    cv::medianBlur(bMinusG, mFilBMinusG, 3);

    cv::Mat demosaicBGR2[3];
    cv::Mat gMinusR = demosaicBGR[1] - demosaicBGR[2];
    cv::Mat gMinusB = demosaicBGR[1] - demosaicBGR[0];
    demosaicBGR2[0] = mFilBMinusG + demosaicBGR[1] - gMinusB;
    demosaicBGR2[1] = demosaicBGR[1];
    demosaicBGR2[2] = mFilRMinusG + demosaicBGR[1] - gMinusR;

    cv::Mat demosaic2, diff2;
    cv::merge(demosaicBGR2, 3, demosaic2);

    return demosaic2;
}

void fillZero4BlueChannel(cv::Mat &blue) {
    for (int row = 0; row < blue.rows; row++) {
        uchar *pointer = blue.ptr(row);
        if (row % 2 == 0) { // set all odd column to 0
            for (int column = 0; column < blue.cols; column++) {
                pointer[column] = 0;
            }
        } else {
            for (int column = 0; column < blue.cols; column++) {
                if (column % 2 == 0) { // set 0 in all even row - odd column
                    pointer[column] = 0;
                }
            }
        }
    }
}

void fillZero4GreenChannel(cv::Mat &green) {
    // set all "?"-place to 0 for green channel
    for (int row = 0; row < green.rows; row++) {
        uchar *pointer = green.ptr(row);
        for (int column = 0; column < green.cols; column++) {
            if ((row % 2 == 0 && column % 2 == 0) || (row % 2 != 0 && column % 2 != 0)) {
                pointer[column] = 0;
            }
        }
    }
}

void fillZero4RedChannel(cv::Mat &red) {
    // set all "?"-place to 0 for red channel
    for (int row = 0; row < red.rows; row++) {
        uchar *pointer = red.ptr(row);
        if (row % 2 != 0) { // set all even column to 0
            for (int column = 0; column < red.cols; column++) {
                pointer[column] = 0;
            }
        } else {
            for (int column = 0; column < red.cols; column++) {
                if (column % 2 != 0) { // set 0 in all odd row - even column
                    pointer[column] = 0;
                }
            }
        }
    }
}

cv::Mat simpleLinearInterpolation4BR(cv::Mat &br) {
    // kernel_1 is for the 1st step of blue and red channel
    float kernel_data_1[9] = {0.25, 0, 0.25,
                              0, 1, 0,
                              0.25, 0, 0.25};
    cv::Mat kernel_1 = cv::Mat(3, 3, CV_32F, kernel_data_1);

    // kernel_2 is for (green channel and) the 2nd step of blue and red channel
    float kernel_data_2[9] = {0, 0.25, 0,
                              0.25, 1, 0.25,
                              0, 0.25, 0};
    cv::Mat kernel_2 = cv::Mat(3, 3, CV_32F, kernel_data_2);

    cv::Point anchor = cv::Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    cv::Mat temp;

    cv::filter2D(br, temp, ddepth, kernel_1, anchor, delta, cv::BORDER_DEFAULT);
    cv::filter2D(temp, br, ddepth, kernel_2, anchor, delta, cv::BORDER_DEFAULT);

    return br;
}

cv::Mat simpleLinearInterpolation4G(cv::Mat &green) {
    // kernel_2 is for green channel (and the 2nd step of blue and red channel)
    float kernel_data_2[9] = {0, 0.25, 0,
                              0.25, 1, 0.25,
                              0, 0.25, 0};
    cv::Mat kernel_2 = cv::Mat(3, 3, CV_32F, kernel_data_2);

    cv::Point anchor = cv::Point(-1, -1);
    double delta = 0;
    int ddepth = -1;
    cv::Mat green_result;

    cv::filter2D(green, green_result, ddepth, kernel_2, anchor, delta, cv::BORDER_DEFAULT);
    return green_result;
}
