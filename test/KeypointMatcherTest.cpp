#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

#include "keypointmatcher.h"
TEST_CASE("computeThreeMaximaFromRotationHistogram", "[keypointMatcher]") {
    int i1, i2, i3;

    std::vector<std::vector<int>> histo{{1, 1, 1, 1, 1}, {1}, {1}, {1, 1}};
    computeThreeMaximaFromRotationHistogram(histo, i1, i2, i3);

    REQUIRE(i1 == 0);
    REQUIRE(i2 == 3);
    REQUIRE(i3 == 1);


    std::vector<std::vector<int>> histo1{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                         {1}, {1}, {1, 1}};
    computeThreeMaximaFromRotationHistogram(histo1, i1, i2, i3);

    REQUIRE(i1 == 0);
    REQUIRE(i2 == 3);
    REQUIRE(i3 == -1);


    std::vector<std::vector<int>> histo2{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                         {1}, {1}, {1}};
    computeThreeMaximaFromRotationHistogram(histo2, i1, i2, i3);

    REQUIRE(i1 == 0);
    REQUIRE(i2 == -1);
    REQUIRE(i3 == -1);
}

TEST_CASE("computeDescriptorDistance", "[keypointMatcher]") {
    std::vector<uint32_t> d1(8, 0xFFFFFFFF);
    cv::Mat desc1(4, 1, CV_32SC1, d1.data());

    std::vector<uint32_t> d2(8, 0x0);
    cv::Mat desc2(4, 1, CV_32SC1, d2.data());

    int dist = computeDescriptorDistance(desc1, desc2);
    REQUIRE(dist == 256);
}
