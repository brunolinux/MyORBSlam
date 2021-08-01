#ifndef PYRAMIDLEVELEXTRACTOR_H
#define PYRAMIDLEVELEXTRACTOR_H

#include <opencv2/core.hpp>

std::vector<int> constructCircleXMaxVec(int radius);

class PyramidLevelExtractor {
public:
    constexpr static int EDGE_THRESHOLD = 19;
    constexpr static int CELL_DEFAULT_SIZE = 30;
    constexpr static int PATCH_SIZE = 31;
    constexpr static int HALF_PATCH_SIZE = 15;
    constexpr static int FAST_RADIUS = 3;

    PyramidLevelExtractor(int _level, float _scale, int _num_features,
                          int _norm_threshold, int _min_threshold);

    std::vector<cv::KeyPoint> computeKeypoint(const cv::Mat &image);

private:
    enum Axis {
        ROW = 0,
        COL
    };

    void constructInnerBorder(cv::Size image_size, int border_size);
    void constructCellInfo(int cell_default_size);
    bool calcCellPosition(int index, Axis dir, int &begin, int &end);

    std::vector<cv::KeyPoint> computeRawKeypoints(const cv::Mat& image);
    std::vector<cv::KeyPoint> distributeQuadTree(const std::vector<cv::KeyPoint>& raw_keypoints);

    void computeOrientation(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

    int level;
    float scale;
    int numFeatures;

    int normThreshold;
    int minThreshold;

    static std::vector<int> xMaxVec;

    cv::Point border_ul;
    cv::Point border_br;

    cv::Size cell_size;
    cv::Size cell_num;
};


#endif // PYRAMIDLEVELEXTRACTOR_H
