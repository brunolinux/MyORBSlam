#ifndef KEYPOINTMATCHER_H
#define KEYPOINTMATCHER_H


#include <vector>
#include <opencv2/core.hpp>

class Frame;

struct Match
{
    int src_idx;
    int dst_idx;
};

struct ShortestDistanceInfo
{
    int best_dist = INT_MAX;
    int second_dist = INT_MAX;
    int index = -1;
};

void computeThreeMaximaFromRotationHistogram(const std::vector<std::vector<int>> &histogram, int &ind1, int &ind2, int &ind3);
int computeDescriptorDistance(const cv::Mat &a, const cv::Mat &b);


class KeypointMatcher
{
public:
    KeypointMatcher(float nn_ratio, bool check_orientation);

    // Matching for the Map Initialization (only used in the monocular case)
    std::vector<Match> SearchForInitialization(Frame *F1, Frame *F2, int windowSize=10);

private:
    ShortestDistanceInfo getBestMatchIndex(Frame *F1, Frame *F2, int index1, int windowSize);

    std::vector<std::vector<int>> getRotationHistogram(Frame *F1, Frame *F2, std::vector<int>& matches1to2);


    static constexpr int TH_HIGH = 100;
    static constexpr int TH_LOW = 50;
    static constexpr int HISTO_BINS = 30;
    static constexpr float HISTO_FACTOR = HISTO_BINS / 360.f;

    float m_nnRatio;
    bool m_isOrientationChecking;
};

#endif // KEYPOINTMATCHER_H
