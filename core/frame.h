#ifndef FRAME_H
#define FRAME_H

#include <opencv2/core.hpp>

class AbstractExtractor;

class Frame
{
public:
    Frame(const cv::Mat& img, AbstractExtractor* extractor);
    ~Frame();

    Frame(const Frame& rh) = default;
    Frame& operator=(const Frame& rh) = default;

    Frame(Frame&& rh);
    Frame &operator=(Frame&& rh);

    static void setCameraIntrinsicMatrix(const cv::Mat& K, const cv::Mat& distortion_coeff);
private:
    void undistortionKeypoints(const std::vector<cv::KeyPoint>& kps);
    void assignKeypoint2Grid();

    static void computeImageBounds(cv::Size image_size);

    static cv::Mat m_K;
    static cv::Mat m_distortion;
    static bool m_isDistortionExist;

    static float m_minX;
    static float m_maxX;
    static float m_minY;
    static float m_maxY;

    static int64_t m_globalIndex;

    AbstractExtractor *m_extractor;

    int64_t m_index;
    int64_t m_timestamp;
    cv::Mat m_image;
    std::vector<cv::KeyPoint> m_kps;
    cv::Mat m_descs;
};

#endif // FRAME_H
