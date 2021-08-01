#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include "abstractextractor.h"
#include "pyramidlevelextractor.h"
#include <vector>


class ORBExtractor : public AbstractExtractor
{
public:
    ORBExtractor(int _num_features, float _scale_factor, int _num_levels,
                 int _normal_thre, int _min_thre);

    virtual AbstractExtractor *loadFromFile(const std::string &file_name) override;
    virtual void saveToFile(const std::string &file_name) const override;

    virtual void operator()(cv::InputArray image, cv::InputArray mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray descriptors) override;

private:
    std::vector<cv::Mat> computePyramidImages(const cv::Mat &image, int border_size);

    void computeKeypointsAndDescriptor(const cv::Mat &image,
                                       std::vector<cv::KeyPoint> &_keypoints,
                                       cv::OutputArray _descriptors);
    void computeORBDescriptor(const cv::Mat &image, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);


    void constructParameters();
    void constructPyramidScaleParameters();
    void constructFeatureNumbersPerLevel();
    void constructPattern();

    int m_numFeatures;
    double m_scaleFactor;
    int m_numLevels;
    int m_normalThreFast;
    int m_minThreFast;

    std::vector<float> m_scaleLevelVec;
    std::vector<float> m_invScaleLevelVec;
    std::vector<float> m_sigma2LevelVec;
    std::vector<float> m_invSigma2LevelVec;

    std::vector<int> m_numFeatureLevelVec;

    std::vector<PyramidLevelExtractor> m_pyramidExtractorVec;

    std::vector<cv::Point> m_pattern;
};

#endif // ORBEXTRACTOR_H
