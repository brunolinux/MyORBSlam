#ifndef ABSTRACTCAMERA_H
#define ABSTRACTCAMERA_H

#include "../frame.h"

class AbstractExtractor;


class AbstractCamera
{
public:
    AbstractCamera() = default;
    virtual ~AbstractCamera() = default;

    virtual std::vector<Frame> getNewestFrame() = 0;
protected:
    AbstractExtractor *m_extractor;
};

#endif // ABSTRACTCAMERA_H
