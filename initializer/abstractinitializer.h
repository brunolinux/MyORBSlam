#ifndef ABSTRACTINITIALIZER_H
#define ABSTRACTINITIALIZER_H


#include <opencv2/core.hpp>
#include <memory>
#include "frame.h"

class AbstractInitializer
{
public:
    AbstractInitializer() = default;
    virtual ~AbstractInitializer() = default;

    virtual bool initialize(std::shared_ptr<Frame> frame) = 0;

protected:
    cv::Mat Rcw;
    cv::Mat tcw;

};

#endif // ABSTRACTINITIALIZER_H
