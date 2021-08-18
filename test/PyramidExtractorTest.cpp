#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>
#include "pyramidlevelextractor.h"


TEST_CASE("PyramidLevelExtractor", "[PyramidLevelExtractor]") {
    PyramidLevelExtractor extractor(0, 1., 300, 20, 7);

    cv::Mat image = cv::imread("../data/1305031452.791720.png", 0);

    auto keypoints = extractor.computeKeypoint(image);

    cv::Mat vis_mat;
    cv::drawKeypoints(image, keypoints, vis_mat, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("keypoints", vis_mat);
    int key = cv::waitKey(0);
    if (key == 27) {
        return;
    }
}
