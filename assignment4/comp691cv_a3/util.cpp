#include "util.h"

using namespace std;
using namespace cv;

void show_window(char const *const name, Mat const &mat) {
    namedWindow(name, WINDOW_AUTOSIZE);
    imshow(name, mat);
}

float fwrap(float x, float divisor) {
    if (divisor == 0.0f) { return x; }
    float m = x - divisor * floor(x / divisor);

    //Handle boundary cases resulted from floating-point cut off:
    if (divisor > 0.0f) {
        //Modulo range: [0..divisor)
        if (m >= divisor) {
            //mod(-1e-16, 360.0): m= 360.0
            return 0.0f;
        }
        if (m < 0.0f) {
            if (divisor + m == divisor) {
                return 0.0f; //Just in case
            } else {
                //mod(106.81415022205296, _TWO_PI): m= -1.421e-14 
                return divisor + m;
            }
        }
    } else {
        //Modulo range: (divisor..0]
        if (m <= divisor) {
            //mod(1e-16, -360.0): m= -360.0
            return 0.0f;
        } else {
            //mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14 
            return divisor + m;
        }
    }

    return m;
}

Mat gaussianKernelNormalized(int half_size, float sigma, bool make_odd) {
    int size = make_odd ?
               half_size * 2 + 1 :
               half_size * 2;
    Mat output(size, size, CV_32F);

    int const max = output.rows;
    float const denominator = 2 * sigma * sigma;

    float total_weight = 0.0f;
    for (int r = 0; r < max; ++r) {
        int const r_val = make_odd ?
                          r - half_size :
                          (r < 0 ? r - half_size : r + 1);
        for (int c = 0; c < max; ++c) {
            int const c_val = make_odd ?
                              c - half_size :
                              (c < 0 ? c - half_size : c + 1);

            float const numerator = r_val * r_val + c_val * c_val;
            float const val = exp(-numerator / denominator);
            output.at<float>(r, c) = val;

            total_weight += val;
        }
    }
    for (int r = 0; r < max; ++r) {
        for (int c = 0; c < max; ++c) {
            output.at<float>(r, c) /= total_weight;
        }
    }
    return output;
}

Mat derivative_x(Mat const &input) {
    int const half_size = 1;
    int const size = half_size * 2 + 1;
    Mat output;
    Mat kernel(size, 3, CV_32F);
    for (int r = 0; r < size; ++r) {
        kernel.at<float>(r, 0) = -1.0f;
        kernel.at<float>(r, 1) = 0.0f;
        kernel.at<float>(r, 2) = 1.0f;
    }
    filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
    //derivative is zero at borders
    for (int c = 0; c < output.cols; ++c) {
        output.at<float>(0, c) = 0.0f;
        output.at<float>(output.rows - 1, c) = 0.0f;
    }
    for (int r = 0; r < output.rows; ++r) {
        output.at<float>(r, 0) = 0.0f;
        output.at<float>(r, output.cols - 1) = 0.0f;
    }
    return output;
}

Mat derivative_y(Mat const &input) {
    int const half_size = 1;
    int const size = half_size * 2 + 1;
    Mat output;
    Mat kernel(3, size, CV_32F);
    for (int c = 0; c < size; ++c) {
        kernel.at<float>(0, c) = -1.0f;
        kernel.at<float>(1, c) = 0.0f;
        kernel.at<float>(2, c) = 1.0f;
    }
    filter2D(input, output, -1, kernel, cv::Point(-1, -1), 0, BORDER_CONSTANT);
    //derivative is zero at borders
    for (int c = 0; c < output.cols; ++c) {
        output.at<float>(0, c) = 0.0f;
        output.at<float>(output.rows - 1, c) = 0.0f;
    }
    for (int r = 0; r < output.rows; ++r) {
        output.at<float>(r, 0) = 0.0f;
        output.at<float>(r, output.cols - 1) = 0.0f;
    }
    return output;
}

bool is_local_maximum(Mat const &input, int check_half_size, int row, int col) {
    float response = input.at<float>(row, col);
    for (int offset_r = -check_half_size; offset_r <= check_half_size; ++offset_r) {
        int check_r = row + offset_r;
        if (check_r < 0 || check_r >= input.rows) {
            continue;
        }
        for (int offset_c = -check_half_size; offset_c <= check_half_size; ++offset_c) {
            int check_c = col + offset_c;
            if (check_c < 0 || check_c >= input.cols) {
                continue;
            }
            if (check_r == row && check_c == col) {
                continue; //Don't check self
            }
            float check_response = input.at<float>(check_r, check_c);
            if (check_response >= response) {
                //We are not a local maximum
                return false;
            }
        }
    }
    return true;
}

//[-b +- sqrt(b^2-4ac)]/2a
void quadratic_roots(double a, double b, double c, float &r1, float &r2) {
    double sqrt_discriminant = sqrt(b * b - 4 * a * c);
    double denominator = 2 * a;
    r1 = (float) ((-b + sqrt_discriminant) / denominator);
    r2 = (float) ((-b - sqrt_discriminant) / denominator);
}

void eigen2x2(float a, float b, float c, float d, float &e1, float &e2) {
    float p_b = -(a + d);
    float p_c = a * d - b * c;

    quadratic_roots(1, p_b, p_c, e1, e2);
}

//location of top-left of 4x4 grid = r, c
//dx and dy must have same rows and cols, have 1 float channel
Vec<float, 8>
histogram4x4(Mat const &dx_mat, Mat const &dy_mat, Mat const &gaussian, int gaussian_r, int gaussian_c, int r, int c,
             float const rad) {
    Vec<float, 8> result = 0; //zero initialize...
    int const max_r = r + 4;
    int const max_c = c + 4;
    for (int cur_r = r; cur_r < max_r; ++cur_r) {
        if (cur_r < 0 || cur_r >= dx_mat.rows) {
            continue;
        }
        for (int cur_c = c; cur_c < max_c; ++cur_c) {
            if (cur_c < 0 || cur_c >= dx_mat.cols) {
                continue;
            }
            float dx = dx_mat.at<float>(cur_r, cur_c);
            float dy = dy_mat.at<float>(cur_r, cur_c);
            float mag = sqrt(dx * dx + dy * dy);
            float dir = atan2(dy, dx); //Range is [-pi,+pi]
            dir -= rad; //Range is [-2*pi, +2*pi]
            dir = fwrap(dir, 2 * PI); //[0.0f, 2*pi) ?
            float const RAD_PER_BIN = PI / 4.0f;
            int bin = dir / RAD_PER_BIN;
            //At this point, bin could potentially be 8,
            //giving as an out of index error, because
            //2pi/(pi/4) = 2pi * 4 / pi = 8pi/pi = 8 <-- Out of index!
            if (bin > 7) {
                bin = 7;
            }

            float weighted_mag = gaussian.at<float>(
                    gaussian_r + cur_r - r,
                    gaussian_c + cur_c - c
            );
            result[bin] += weighted_mag;
        }
    }
    return result;
}

float magnitude(Vec<float, 128> const &v) {
    float mag_sqr = 0.0f;
    for (int i = 0; i < 128; ++i) {
        mag_sqr += v[i] * v[i];
    }
    return sqrt(mag_sqr);
}

void normalize(Vec<float, 128> &v) {
    float mag = magnitude(v);
    v /= mag;
}

bool truncate_above(Vec<float, 128> &v, float threshold) {
    bool truncated = false;
    for (int i = 0; i < 128; ++i) {
        if (v[i] > threshold) {
            v[i] = threshold;
            truncated = true;
        }
    }
    return truncated;
}

bool hasnan(Vec<float, 128> &v) {
    for (int i = 0; i < 128; ++i) {
        if (isnan(v[i])) {
            return true;
        }
    }
    return false;
}

bool is_zero(Vec<float, 128> &v) {
    for (int i = 0; i < 128; ++i) {
        if (v[i] != 0.0f) {
            return false;
        }
    }
    return true;
}

Vec<float, 128> histogram128_from_4x4arr(Vec<float, 8> (*histogram_arr)[4]) {
    Vec<float, 128> result;
    //Copy values of 4x4 histogram to result
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            Vec<float, 8> histogram = histogram_arr[r][c];
            int index_1d = r * 4 + c;
            for (int i = 0; i < 8; ++i) {
                result[index_1d + i] = histogram[i];
                if (isnan(histogram[i])) {
                    throw "wtf";
                }
            }
        }
    }
    if (is_zero(result)) {
        return result; //This is quite wtf-y
    }
    normalize(result);
    if (hasnan(result)) {
        throw "wtf";
    }
    Vec<float, 128> copy = result;
    if (truncate_above(result)) {
        if (is_zero(result)) {
            return copy; //This is quite wtf-y
        }
        /*normalize(result);
        if (hasnan(result)) {
            throw "wtf";
        }*/
    }
    return result;
}

//keypoint location = r, c
//gaussian has to be a 16x16 kernel
Vec<float, 128> histogram128(Mat const &dx, Mat const &dy, Mat const &gaussian, int r, int c, float rad) {
    Vec<float, 8> histogram_arr[4][4];
    //We want 16 of these 4x4 histograms...
    for (int cur_r = 0; cur_r < 4; ++cur_r) {
        int histogram_r = r + (cur_r - 2) * 4;
        int gaussian_r = cur_r * 4;
        for (int cur_c = 0; cur_c < 4; ++cur_c) {
            int histogram_c = c + (cur_c - 2) * 4;
            int gaussian_c = cur_c * 4;
            histogram_arr[cur_r][cur_c] = histogram4x4(
                    dx, dy, gaussian,
                    gaussian_r, gaussian_c,
                    histogram_r, histogram_c,
                    rad
            );
        }
    }
    return histogram128_from_4x4arr(histogram_arr);
}

//Returns a rad rotation, [-pi, pi)
void
dominant_orientations(Mat const &dx_mat, Mat const &dy_mat, int r, int c, float sigma, vector<float> &rad_results) {
    int const half_size = 4;// (int)(sigma*1.5f);
    int const size = 2 * half_size + 1;
    float const RAD_PER_BIN = PI / 18.0f;
    Mat const gaussian = gaussianKernelNormalized(half_size, sigma * 1.5f);

    Vec<float, 36> histogram;
    for (int cur_r = r - half_size; cur_r < r + half_size; ++cur_r) {
        if (cur_r < 0 || cur_r >= dx_mat.rows) {
            continue;
        }
        for (int cur_c = c - half_size; cur_c < c + half_size; ++cur_c) {
            if (cur_c < 0 || cur_c >= dx_mat.cols) {
                continue;
            }
            float dx = dx_mat.at<float>(cur_r, cur_c);
            float dy = dy_mat.at<float>(cur_r, cur_c);
            float mag = sqrt(dx * dx + dy * dy);
            float dir = atan2(dy, dx); //Range is [-pi,+pi]
            dir = fwrap(dir, 2 * PI); //[0.0f, 2*pi) ?
            int bin = dir / RAD_PER_BIN;
            //At this point, bin could potentially be 36,
            //giving as an out of index error, because
            //2pi/(pi/18) = 2pi * 18 / pi = 36pi/pi = 36 <-- Out of index!
            if (bin > 35) {
                bin = 35;
            }
            histogram[bin] += mag * gaussian.at<float>(cur_r - r + half_size, cur_c - c + half_size);
        }
    }

    // find the dominant one
    int dominant_bin = -1;
    float dominant_mag = 0.0f;
    for (int i = 0; i < 36; ++i) {
        float mag = histogram[i];
        if (mag > dominant_mag) {
            dominant_bin = i;
            dominant_mag = mag;
        }
    }

    if (dominant_bin < 0) {
        return; //No keypoint for you!
    }
    for (int i = 0; i < 36; ++i) {
        float mag = histogram[i];
        if (mag / dominant_mag > 0.8f) {
            rad_results.push_back(i * RAD_PER_BIN);
        }
    }
}

bool filter_matches(vector<match> &intermediate, vector<match> &result, bool(*pred)(match, match)) {
    if (intermediate.size() == 0) {
        return false;
    }
    match best = intermediate[0];
    match next_best = {-1, -1, INT_MAX};
    for (int i = 1; i < intermediate.size(); ++i) {
        match cur = intermediate[i];
        if (!pred(best, cur)) {
            continue;
        }
        if (cur.distance < best.distance) {
            next_best = best;
            best = cur;
        }
    }
    if (next_best.distance < INT_MAX) {
        if (best.distance / next_best.distance < 0.8f) {
            result.push_back(best);
        }
    } else {
        result.push_back(best);
    }
    intermediate.erase(remove_if(intermediate.begin(), intermediate.end(), [best, pred](match m) {
        return pred(best, m);
    }), intermediate.end());
    return true;
}

void match_features(
        vector<feature> const &features_a, vector<feature> const &features_b,
        float max_distance_threshold, bool do_filter_matches,
        vector<match> &result
) {
    float min_dist = INFINITY;
    float max_dist = -INFINITY;
    vector<match> intermediate_result;
    for (int a = 0; a < features_a.size(); ++a) {
        feature f_a = features_a[a];
        match best_match = {-1, -1, INT_MAX};
        match next_best_match = {-1, -1, INT_MAX};
        if (a % 100 == 0) {
            cout << "Progress: " << a << "/" << features_a.size() << endl;
        }
        for (int b = 0; b < features_b.size(); ++b) {
            feature f_b = features_b[b];
            // SSD
            auto diff = f_a.f - f_b.f;
            diff = diff.mul(diff);
            float distance = 0.0f;
            for (int i = 0; i < 128; ++i) {
                distance += diff[i];
            }

            // the min_dist and max_dist are only for debugging
            if (distance < min_dist) {
                min_dist = distance;
            }
            if (distance > max_dist) {
                max_dist = distance;
            }

            if (distance < max_distance_threshold) {
                match m = {
                        a,
                        b,
                        distance,
                        f_a.p.r,
                        f_a.p.c,
                        f_b.p.r,
                        f_b.p.c
                };
                if (m.distance < best_match.distance) {
                    next_best_match = best_match;
                    best_match = m;
                }
            }
        }
        if (best_match.distance < INT_MAX) {
            if (next_best_match.distance < INT_MAX) {
                if (best_match.distance / next_best_match.distance < 0.8f) {
                    //The next best match isn't *quite* as good as this best match
                    intermediate_result.push_back(best_match);
                }
            } else {
                //No contest, this is the best match
                intermediate_result.push_back(best_match);
            }
        }
    }
    if (do_filter_matches) {
        vector<match> i_b;
        while (filter_matches(intermediate_result, i_b, [](match a, match b) {
            return a.b_r == b.b_r && a.b_c == b.b_c;
        })) {
        };
        while (filter_matches(i_b, result, [](match a, match b) {
            return a.a_r == b.a_r && a.a_c == b.a_c;
        })) {
        };
    } else {
        result = intermediate_result;
    }

    // just for debugging
    cout << "min dist: " << min_dist << endl;
    cout << "max dist: " << max_dist << endl;
}