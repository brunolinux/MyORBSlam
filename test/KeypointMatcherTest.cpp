#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

#include "keypointmatcher.h"
#include "orbextractor.h"
#include "frame.h"

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

TEST_CASE("matcher test", "[keypointMatcher]") {
    ORBExtractor extractor(1000, 1.2, 8, 20, 7);

    cv::Mat image1 = cv::imread("../data/1305031452.791720.png", 0);
    std::shared_ptr<Frame> frame_reference = std::make_shared<Frame>(image1, &extractor);

    cv::Mat image2 = cv::imread("../data/1305031452.823674.png", 0);
    std::shared_ptr<Frame> frame_target = std::make_shared<Frame>(image2, &extractor);

    KeypointMatcher matcher(0.9, true);
    auto matches = matcher.SearchForInitialization(frame_reference.get(), frame_target.get(), 100);

    std::vector<cv::DMatch> dmatches(matches.size());
    for (size_t i = 0; i < matches.size(); i ++) {
        dmatches[i].queryIdx = matches[i].src_idx;
        dmatches[i].trainIdx = matches[i].dst_idx;
    }

    cv::Mat vis_mat;
    cv::drawMatches(image1, frame_reference->getKeypoints(), image2, frame_target->getKeypoints(), dmatches, vis_mat);

    cv::imshow("match result", vis_mat);
    cv::waitKey(0);
}
