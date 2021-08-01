#ifndef ABSTRACTEXTRACTOR_H
#define ABSTRACTEXTRACTOR_H

#include <string>
#include <opencv2/core.hpp>

class AbstractExtractor
{
public:
    AbstractExtractor() = default;
    virtual ~AbstractExtractor() = default;

    virtual AbstractExtractor *loadFromFile(const std::string &file_name) = 0;
    virtual void saveToFile(const std::string &file_name) const = 0;

    virtual void operator()(cv::InputArray image, cv::InputArray mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray descriptors) = 0;
};

#endif // ABSTRACTEXTRACTOR_H
