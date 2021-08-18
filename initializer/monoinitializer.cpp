#include "monoinitializer.h"

constexpr size_t MIN_KEYPOINT_NUM = 100;

MonoInitializer::MonoInitializer()
    :m_state(InitState::NOInit)
{

}

bool MonoInitializer::initialize(const Frame &frame)
{
    if (m_state == InitState::NOInit) {
        if (checkFrame(frame)) {
            m_reference = frame;
            m_state = InitState::ReferencePassed;
        }

        return false;
    } else if (m_state == InitState::ReferencePassed) {
        if (checkFrame(frame)) {
            m_target = frame;

            if (checkFrameMatching()) {
                // TODO
                m_state = InitState::TargetPassed;

                return true;
            }
        }

        m_state = InitState::NOInit;
        return false;
    } else {
        return true;
    }
}

bool MonoInitializer::checkFrame(const Frame &frame) const
{
    if (frame.kps.size() < MIN_KEYPOINT_NUM) {
        return false;
    } else {
        return true;
    }
}

bool MonoInitializer::checkFrameMatching()
{
    return false;
}
