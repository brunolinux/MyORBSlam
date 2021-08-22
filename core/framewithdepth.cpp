#include "framewithdepth.h"

using std::vector;

float FrameWithDepth::m_baseline = 0.f;

FrameWithDepth::FrameWithDepth(const cv::Mat &img, AbstractExtractor *extractor)
    :Frame(img, extractor)
{

}

void FrameWithDepth::computeStereoFromRGBD(const cv::Mat &image_depth)
{
    CV_Assert(m_baseline > 0.f);

    size_t N = m_kps.size();
    m_rightXValues = vector<float>(N, -1);
    m_depths = vector<float>(N,-1);

    for(size_t i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = m_kps[i];
        const cv::KeyPoint &kpU = m_kpsUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = image_depth.at<float>(v, u);

        if(d > 0)
        {
            m_depths[i] = d;
            m_rightXValues[i] = kpU.pt.x- m_baseline/d;
        }
    }
}

void FrameWithDepth::computeStereoMatches(const Frame &left_frame)
{
    // TODO
}

void FrameWithDepth::setBaseline(float length)
{
    m_baseline = length;
}
