#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void simpleHybrid(Mat &input1, Mat &input2,
                  int sizex1, int sizey1,
                  int sizex2, int sizey2,
                  double sigma1, double sigma2,
                  Mat &hybrid) {
    Mat output1, output2;
    GaussianBlur(input1, output1, Size(sizex1, sizey1), sigma1);
    Mat temp;
    GaussianBlur(input2, temp, Size(sizex2, sizey2), sigma2);
    output2 = input2 - temp;
    //offset(output2, 128);
    hybrid = output1 + output2;

    /*namedWindow("offset", WINDOW_AUTOSIZE);
    imshow("offset", output2);
    waitKey(0);*/
    namedWindow("input1", WINDOW_AUTOSIZE);
    imshow("input1", input1);
    waitKey(0);
    namedWindow("GaussianBlur 1", WINDOW_AUTOSIZE);
    imshow("GaussianBlur 1", output1);
    waitKey(0);
    namedWindow("input2", WINDOW_AUTOSIZE);
    imshow("input2", input2);
    waitKey(0);
    namedWindow("GaussianBlur 2", WINDOW_AUTOSIZE);
    imshow("GaussianBlur 2", output2);
    waitKey(0);
    namedWindow("hybrid", WINDOW_AUTOSIZE);
    imshow("hybrid", hybrid);
    waitKey(0);
}

void offset(Mat &input, int offset) {
    for (int row = 0; row < input.rows; row++) {
        uchar *pointer = input.ptr(row);
        for (int column = 0; column < input.cols * 3; column += 3) {
            pointer[column] += offset;
            pointer[column + 1] += offset;
            pointer[column + 2] += offset;
        }
    }
}

void GaussianKernel(int sizex, int sizey, double sigma, Mat &kernel) {
    kernel = Mat(sizey, sizex, CV_64F);
    float centerX = (sizex - 1) / 2;
    float centerY = (sizey - 1) / 2;
    for (int row = 0; row < sizey; row++) {
        for (int column = 0; column < sizex; column++) {
            double element = 1 / sigma / sigma / 2 / M_PI
                             * exp(-1.0 / 2 *
                                   ((centerX - column) * (centerX - column) + (centerY - row) * (centerY - row)) /
                                   sigma / sigma);
            kernel.at<double>(row, column) = element;
        }
    }
}

//Assumes input_image is int, do not call directly..
//Assumes kernel size is odd in both dimensions
template<int Ch> // using template to avoid repeate in coding for different number of channels
Vec<uchar, Ch> weightedAverage(Mat const &input_image, Mat const &kernel, int const input_r_origin, int const input_c_origin) {
    int const input_r_start = input_r_origin - ((kernel.rows - 1) / 2);
    int const input_c_start = input_c_origin - ((kernel.cols - 1) / 2);
    Vec<double, Ch> total;
    double total_weight = 0.0f;
    for (int r = 0; r < kernel.rows; ++r) {
        int const input_r = input_r_start + r;
        // to check if the element of the kernel is outside of image border - row
        if (input_r < 0 || input_r >= input_image.rows) {
            continue;
        }
        for (int c = 0; c < kernel.cols; ++c) {
            int const input_c = input_c_start + c;
            // to check if the element of the kernel is outside of image border - column
            if (input_c < 0 || input_c >= input_image.cols) {
                continue;
            }
            Vec<uchar, Ch> input_pixel = input_image.at<Vec<uchar, Ch>>(input_r, input_c);
            double w = kernel.at<double>(kernel.rows - 1 - r, kernel.cols - 1 - c);
            total_weight += w;
            for (int i = 0; i < Ch; ++i) {
                total[i] += input_pixel[i] * w;
            }
        }
    }
    return total / total_weight;
}

// dealing with input with all possible number of channels
void filtering(Mat const &input_image, Mat const &kernel, Mat &output_image) {
    if (kernel.rows % 2 == 0) {
        cerr << "Kernel has even number of rows: " << kernel.rows << endl;
        return;
    }
    if (kernel.cols % 2 == 0) {
        cerr << "Kernel has even number of cols: " << kernel.cols << endl;
        return;
    }
    if (input_image.rows != output_image.rows) {
        cerr << "Input rows != Output rows: " << input_image.rows << " != " << output_image.rows << endl;
        return;
    }
    if (input_image.cols != output_image.cols) {
        cerr << "Input cols != Output cols: " << input_image.cols << " != " << output_image.cols << endl;
        return;
    }
    if (input_image.type() != output_image.type()) {
        cerr << "Input type() != Output type(): " << input_image.type() << " != " << output_image.type() << endl;
        return;
    }
    if (input_image.channels() != output_image.channels()) {
        cerr << "Input channels() != Output channels(): " << input_image.channels() << " != " << output_image.channels()
             << endl;
        return;
    }
    if (kernel.type() != CV_64F) {
        cerr << "kernel type() != CV_64F: " << kernel.type() << endl;
        return;
    }
    if (kernel.channels() != 1) {
        cerr << "kernel channels() != 1: " << kernel.channels() << endl;
        return;
    }
    for (int r = 0; r < input_image.rows; ++r) {
        for (int c = 0; c < input_image.cols; ++c) {
            switch (input_image.channels()) {
                case 1:
                    output_image.at<uchar>(r, c) = weightedAverage<1>(input_image, kernel, r, c)[0];
                    break;
                case 2:
                    output_image.at<Vec2b>(r, c) = weightedAverage<2>(input_image, kernel, r, c);
                    break;
                case 3:
                    output_image.at<Vec3b>(r, c) = weightedAverage<3>(input_image, kernel, r, c);
                    break;
                case 4:
                    output_image.at<Vec4b>(r, c) = weightedAverage<4>(input_image, kernel, r, c);
                    break;
                default:
                    break;
            }
        }
    }
}

void sobel_diy(Mat &input, Mat &output) {
    float kernelArrayX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float kernelArrayY[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    Mat kernelX = Mat(3, 3, CV_32F, kernelArrayX);
    Mat kernelY = Mat(3, 3, CV_32F, kernelArrayY);
    Mat sobelX, sobelY;
    filter2D(input, sobelX, -1, kernelX);
    filter2D(input, sobelY, -1, kernelY);
    output = 0.5 * sobelX + 0.5 * sobelY;
}

int main() {
    Mat cat_src = imread(
            "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment2/comp691cv-a2/resources/cat.bmp",
            IMREAD_COLOR);
    if (cat_src.empty()) {
        cout << "Could not open or find image";
    }

    Mat dog_src = imread(
            "/Users/m.ding/Documents/concordia/courses/comp691cv/assignment2/comp691cv-a2/resources/dog.bmp",
            IMREAD_COLOR);
    if (dog_src.empty()) {
        cout << "Could not open or find image";
    }

    // part 1
    Mat hybrid;
    simpleHybrid(dog_src, cat_src, 21, 21, 21, 21, 2048, 2048, hybrid);

    Mat cat_dst(cat_src.rows, cat_src.cols, cat_src.type());
    Mat dog_dst(dog_src.rows, dog_src.cols, dog_src.type());


    // part 2
    int const size = 10 * 2 + 1;
    int const sigma = 2048;

    Mat kernel_cat;
    GaussianKernel(size, size, sigma, kernel_cat);
    Mat kernel_temp_cat;
    GaussianKernel(7, 7, sigma, kernel_temp_cat);
    cout<<kernel_temp_cat<<endl;
    filtering(cat_src, kernel_cat, cat_dst);

    cat_dst = cat_src - cat_dst;
    imshow("cat my filtering", cat_dst);
    waitKey(0);

    Mat kernel_dog = getGaussianKernel(size, sigma);
    kernel_dog = kernel_dog * kernel_dog.t();
    GaussianKernel(25, 25, 2048, kernel_dog);
    filtering(dog_src, kernel_dog, dog_dst);
    imshow("dog my filtering", dog_dst);
    waitKey(0);

    Mat hybrid_part2 = cat_dst + dog_dst;
    imshow("hybrid part 2", hybrid_part2);
    waitKey(0);


    // part 3
    Mat dog_kernel;
    GaussianKernel(21, 21, 2048, dog_kernel);
    filtering(dog_src, dog_kernel, dog_dst);

    //sobel operator
    sobel_diy(cat_src, cat_dst);
    namedWindow("my cat sobel", WINDOW_AUTOSIZE);
    imshow("my cat sobel", cat_dst);
    waitKey(0);

    Mat sobelResult = dog_dst + cat_dst;
    namedWindow("hybrid Sobel", WINDOW_AUTOSIZE);
    imshow("hybrid Sobel", sobelResult);
    waitKey(0);

    //DoG
    Mat cat_Gaussian1, cat_Gaussian2;
    Mat cat_kernel_1, cat_kernel_2;
    Mat cat_DoG(cat_src.rows, cat_src.cols, cat_src.type());
    GaussianKernel(21, 21, 1024, cat_kernel_1);
    GaussianKernel(21, 21, 2048, cat_kernel_2);
    filtering(cat_src, cat_kernel_1 - cat_kernel_2, cat_DoG);
    cat_DoG = cat_src - cat_DoG;

    namedWindow("cat DoG", WINDOW_AUTOSIZE);
    imshow("cat DoG", cat_DoG);
    waitKey(0);

    Mat DoGResult = dog_dst + cat_DoG;
    namedWindow("hybrid DoG", WINDOW_AUTOSIZE);
    imshow("hybrid DoG", DoGResult);
    waitKey(0);

    //LoG
    Mat cat_LoG(cat_src.rows, cat_src.cols, cat_src.type());
    Mat kernel_G, temp(cat_src.rows, cat_src.cols, cat_src.type());
    //GaussianKernel(21, 21, 2600, kernel_G);
    //filtering(cat_src, kernel_G, temp);

    GaussianKernel(21, 21, 2048.0 / sqrt(2), cat_kernel_1);
    GaussianKernel(21, 21, sqrt(2) * 2048.0, cat_kernel_2);
    //filtering(temp, cat_kernel_1 - cat_kernel_2, cat_DoG);
    filtering(cat_src, cat_kernel_1 - cat_kernel_2, cat_DoG);
    cat_LoG = cat_src - cat_DoG;

    namedWindow("cat LoG", WINDOW_AUTOSIZE);
    imshow("cat LoG", cat_LoG);
    waitKey(0);

    Mat LoGResult = dog_dst + cat_LoG;
    namedWindow("hybrid LoG", WINDOW_AUTOSIZE);
    imshow("hybrid LoG", LoGResult);
    waitKey(0);


    return 0;
}
