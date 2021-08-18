#ifndef ABSTRACTINITIALIZER_H
#define ABSTRACTINITIALIZER_H


#include <opencv2/core.hpp>
#include "frame.h"

class AbstractInitializer
{
public:
    AbstractInitializer() = default;
    virtual ~AbstractInitializer() = default;

    virtual bool initialize(const Frame& frame) = 0;
};

#endif // ABSTRACTINITIALIZER_H
