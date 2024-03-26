#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>

using namespace cv;

int main() {
    Mat img = imread("C:/Users/User/Desktop/mat.webp");
    if (img.empty()) {
        std::cerr << "Error: Unable to load image" << std::endl;
        return -1;
    }

    Mat grayscale_img = Mat::zeros(img.size(), CV_8UC1);
    Mat sepia_img = img.clone();
    Mat negative_img = img.clone();
    Mat img_p = Mat::zeros(img.size(), CV_8UC1);
    Mat contour_img = Mat::zeros(img.size(), CV_8UC1);


    imshow("Original Image", img);
    waitKey(0);


    {
        {
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    Vec3b pixel = img.at<Vec3b>(i, j);
                    grayscale_img.at<uchar>(i, j) = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
                }
            }
            imshow("Grayscale Image", grayscale_img);
            waitKey(0);
        }

        {
            sepia_img.forEach<Vec3b>([](Vec3b& pixel, const int* position) {
                uchar blue = pixel[0];
                uchar green = pixel[1];
                uchar red = pixel[2];
                pixel[0] = saturate_cast<uchar>(0.272 * red + 0.534 * green + 0.131 * blue);
                pixel[1] = saturate_cast<uchar>(0.349 * red + 0.686 * green + 0.168 * blue);
                pixel[2] = saturate_cast<uchar>(0.393 * red + 0.769 * green + 0.189 * blue);
                });
            imshow("Sepia Image", sepia_img);
            waitKey(0);
        }

        {
            negative_img.forEach<Vec3b>([](Vec3b& pixel, const int* position) {
                pixel[0] = 255 - pixel[0];
                pixel[1] = 255 - pixel[1];
                pixel[2] = 255 - pixel[2];
                });
            imshow("Negative Image", negative_img);
            waitKey(0);
        }

        {
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    Vec3b pixel = img.at<Vec3b>(i, j);
                    img_p.at<uchar>(i, j) = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
                }
            }
            for (int i = 1; i < img_p.rows - 1; ++i) {
                for (int j = 1; j < img_p.cols - 1; ++j) {
                    float gx = img_p.at<uchar>(i + 1, j + 1) + 2 * img_p.at<uchar>(i, j + 1) + img_p.at<uchar>(i - 1, j + 1) - img_p.at<uchar>(i + 1, j - 1) - 2 * img_p.at<uchar>(i, j - 1) - img_p.at<uchar>(i - 1, j - 1);
                    float gy = img_p.at<uchar>(i + 1, j + 1) + 2 * img_p.at<uchar>(i + 1, j) + img_p.at<uchar>(i + 1, j - 1) - img_p.at<uchar>(i - 1, j - 1) - 2 * img_p.at<uchar>(i - 1, j) - img_p.at<uchar>(i - 1, j + 1);
                    contour_img.at<uchar>(i, j) = 255 - sqrt(pow(gx, 2) + pow(gy, 2));
                }
            }
            imshow("Contour Image", contour_img);
            waitKey(0);
        }
    }

    return 0;
}