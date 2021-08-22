#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>

class AbstractExtractor;

class Frame
{
public:
    Frame(const cv::Mat& img, AbstractExtractor* extractor);
    virtual ~Frame() = default;

    static void setCameraIntrinsicMatrix(const cv::Mat& K, const cv::Mat& distortion_coeff);

    size_t getKeypointSize() const;
    const std::vector<cv::KeyPoint>& getUndistortedKeypoints() const;

    std::vector<size_t> getFeatureIndexesInCircle(const cv::Point2f& point, const float  &r, const int minLevel, const int maxLevel) const;


    static cv::Mat getIntrinsicMatrix();
protected:
    void calcImageSizeRelatedValue(cv::Size image_size);

    void undistortionKeypoints();
    void assignKeypoint2Grid();

    static void computeImageBounds(cv::Size image_size);
    static bool getPointGridPixel(const cv::KeyPoint &kp, int &posX, int &posY);

    static cv::Mat m_K;
    static cv::Mat m_distortion;
    static bool m_isDistortionExist;

    static bool m_isImageSizeRelatedValueCalculated;
    static float m_minX;
    static float m_maxX;
    static float m_minY;
    static float m_maxY;
    static float m_gridCellWidthInv;
    static float m_gridCellHeightInv;

    static int64_t m_globalIndex;

    AbstractExtractor *m_extractor;

    int64_t m_index;
    int64_t m_timestamp;
    cv::Mat m_image;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    std::vector<cv::KeyPoint> m_kps;
    std::vector<cv::KeyPoint> m_kpsUn;
    cv::Mat m_descs;

    static constexpr int FRAME_GRID_ROWS = 48;
    static constexpr int FRAME_GRID_COLS = 64;
    std::vector<std::size_t> m_grid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    friend class KeypointMatcher;
};

#endif // FRAME_H
