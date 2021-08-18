#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

#include "orbextractor.h"
TEST_CASE("ORB Extractor", "[ORBExtractor]") {
    ORBExtractor extractor(1000, 1.2, 8, 20, 7);

    cv::Mat image = cv::imread("../data/1305031452.791720.png", 0);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    extractor(image, cv::noArray(), keypoints, descriptors);

    cv::Mat vis_mat;
    cv::drawKeypoints(image, keypoints, vis_mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("ORB keypoints", vis_mat);
    int key = cv::waitKey(0);
    if (key == 27) {
        return;
    }
}
