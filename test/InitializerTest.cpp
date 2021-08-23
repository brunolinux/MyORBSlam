#include <catch2/catch.hpp>
#include <opencv2/opencv.hpp>

#include "monoinitializer.h"
#include "orbextractor.h"
TEST_CASE("MonoInitializer Test", "[MonoInitializer]") {
    ORBExtractor extractor(1000, 1.2, 8, 20, 7);

    cv::Mat K = (cv::Mat_<float>(3, 3)
                 << 517.306408, 0, 318.643040,
                    0, 516.469215, 255.313989,
                    0, 0, 1);

    cv::Mat distortion = (cv::Mat_<float>(5, 1) <<  0.262383, -0.953104, -0.005358, 0.002628, 1.163314);

    Frame::setCameraIntrinsicMatrix(K, distortion);

    cv::Mat image1 = cv::imread("../data/1305031452.791720.png", 0);
    std::shared_ptr<Frame> frame_reference = std::make_shared<Frame>(image1, &extractor);

    cv::Mat image2 = cv::imread("../data/1305031452.791720.png", 0);
    std::shared_ptr<Frame> frame_target = std::make_shared<Frame>(image2, &extractor);

    MonoInitializer initializer(200, 1.0);
    initializer.initialize(frame_reference);
    bool state = initializer.initialize(frame_target);

    REQUIRE(state == true);
}
