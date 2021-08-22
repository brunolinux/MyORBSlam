#ifndef FRAMEWITHDEPTH_H
#define FRAMEWITHDEPTH_H

#include "frame.h"

class FrameWithDepth : public Frame
{
public:
    FrameWithDepth(const cv::Mat& img, AbstractExtractor* extractor);

    void computeStereoFromRGBD(const cv::Mat &image_depth);

    void computeStereoMatches(const Frame& left_frame);

    // note: must be called before
    static void setBaseline(float length);
private:
    static float m_baseline;

    std::vector<float> m_rightXValues;
    std::vector<float> m_depths;
};

#endif // FRAMEWITHDEPTH_H
