#include "frame.h"
#include "abstractextractor.h"
#include <opencv2/opencv.hpp>

int64_t Frame::m_globalIndex = 0;

Frame::Frame(const cv::Mat &img, AbstractExtractor *extractor)
    :m_extractor(extractor), m_image(img)
{
    std::vector<cv::KeyPoint> kps;
    m_extractor->operator()(m_image, cv::noArray(), kps, m_descs);

    undistortionKeypoints(kps);
    assignKeypoint2Grid();

    m_index = m_globalIndex;
    m_globalIndex++;

    {

    }
}

Frame::Frame(Frame &&rh)
{

}

Frame &Frame::operator=(Frame &&rh)
{
    return *this;
}



void Frame::undistortionKeypoints(const std::vector<cv::KeyPoint> &kps)
{
    if (!m_isDistortionExist) {
        m_kps = std::move(kps);
        return;
    }

    int N = static_cast<int>(kps.size());
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i ++) {
        mat.at<float>(i, 0) = kps[i].pt.x;
        mat.at<float>(i, 1) = kps[i].pt.y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, m_K, m_distortion, cv::Mat(), m_K);
    mat = mat.reshape(1);

    m_kps.resize(N);
    for (int i = 0; i < N; i ++) {
        cv::KeyPoint kp = kps[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        m_kps[i] = std::move(kp);
    }
}

void Frame::assignKeypoint2Grid()
{

}

void Frame::computeImageBounds(cv::Size image_size)
{
    if (m_isDistortionExist) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;  mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = image_size.width;  mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;  mat.at<float>(2, 1) = image_size.height;
        mat.at<float>(3, 0) = image_size.width;  mat.at<float>(3, 1) = image_size.height;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, m_K, m_distortion, cv::Mat(), m_K);
        mat = mat.reshape(1);

        m_minX = fmin(mat.at<float>(0, 0), mat.at<float>(2, 0));
        m_maxX = fmax(mat.at<float>(1, 0), mat.at<float>(3, 0));
        m_minY = fmin(mat.at<float>(0, 1), mat.at<float>(1, 1));
        m_maxY = fmax(mat.at<float>(2, 1), mat.at<float>(3, 1));
    } else {
        m_minX = 0.f;
        m_maxX = image_size.width;
        m_minX = 0.f;
        m_maxY = image_size.height;
    }
}

void Frame::setCameraIntrinsicMatrix(const cv::Mat &K, const cv::Mat &distortion_coeff)
{
    m_K = K;
    m_distortion = distortion_coeff;
    if (m_distortion.at<float>(0) == 0.0) {
        m_isDistortionExist = false;
    } else {
        m_isDistortionExist = true;
    }
}


