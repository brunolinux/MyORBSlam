#ifndef MONOINITIALIZER_H
#define MONOINITIALIZER_H

#include "abstractinitializer.h"


class MonoInitializer : public AbstractInitializer
{
public:
    MonoInitializer();

    bool initialize(const Frame& frame) override;

private:
    enum class InitState {
        NOInit,
        ReferencePassed,
        TargetPassed
    };

    bool checkFrame(const Frame& frame) const;
    bool checkFrameMatching();

    InitState m_state;

    Frame m_reference;
    Frame m_target;
};

#endif // MONOINITIALIZER_H
