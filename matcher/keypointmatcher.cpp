#include "keypointmatcher.h"
#include "frame.h"

using std::vector;

KeypointMatcher::KeypointMatcher(float nn_ratio, bool check_orientation)
    :m_nnRatio(nn_ratio), m_isOrientationChecking(check_orientation)
{}

std::vector<Match> KeypointMatcher::SearchForInitialization(Frame *F1, Frame *F2, int windowSize)
{
    std::vector<int> matches1to2 = vector<int>(F1->getKeypointSize(), -1);
    std::vector<int> matches2to1 = vector<int>(F2->getKeypointSize(), -1);

    for (size_t i1 = 0; i1 < F1->getKeypointSize(); i1 ++) {
        ShortestDistanceInfo info = getBestMatchIndex(F1, F2, i1, windowSize);
        if (info.index < 0)
            continue;

        if ((info.best_dist <= TH_LOW) && (info.best_dist < info.second_dist * m_nnRatio) ) {
            if (matches2to1[info.index] >= 0) {
                matches1to2[matches2to1[info.index]] = -1;
            }
            matches1to2[i1] = info.index;
            matches2to1[info.index] = i1;
        }
    }

    if (m_isOrientationChecking) {
        vector<vector<int>> rotHist = getRotationHistogram(F1, F2, matches1to2);

        int ind1=-1, ind2=-1, ind3=-1;

        computeThreeMaximaFromRotationHistogram(rotHist, ind1, ind2, ind3);

        for(int i = 0; i < HISTO_BINS; i++) {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for (size_t j = 0; j < rotHist[i].size(); j++) {
                int idx1 = rotHist[i][j];
                if (matches1to2[idx1] >= 0)
                {
                    matches1to2[idx1] = -1;
                }
            }
        }
    }

    std::vector<Match> matches;
    for (int i1 = 0; i1 < static_cast<int>(matches1to2.size()); i1 ++) {
        if (matches1to2[i1] >= 0) {
            matches.push_back({i1, matches1to2[i1]});
        }
    }
    return matches;
}

ShortestDistanceInfo KeypointMatcher::getBestMatchIndex(Frame *F1, Frame *F2, int index1, int windowSize)
{
    ShortestDistanceInfo info;

    const cv::KeyPoint &kp1 = F1->m_kpsUn[index1];
    int level1 = kp1.octave;
    if (level1 > 0)
        return info;

    vector<size_t> indexes2 = F2->getFeatureIndexesInCircle(kp1.pt, windowSize, level1, level1);

    if (indexes2.empty())
        return info;

    cv::Mat d1 = F1->m_descs.row(index1);
    for (auto i2: indexes2) {
        cv::Mat d2 = F2->m_descs.row(i2);

        int dist = computeDescriptorDistance(d1, d2);

        if(dist < info.best_dist) {
            info.second_dist = info.best_dist;
            info.best_dist = dist;
            info.index = i2;
        } else if(dist < info.second_dist) {
            info.second_dist = dist;
        }
    }

    return info;
}

std::vector<std::vector<int> > KeypointMatcher::getRotationHistogram(Frame *F1, Frame *F2, std::vector<int> &matches1to2)
{
    vector<vector<int>> rotHist(HISTO_BINS);
    for(int i = 0; i < HISTO_BINS; i++)
        rotHist[i].reserve(500);

    for (size_t i1 = 0; i1 < F1->getKeypointSize(); i1 ++) {
        int i2 = matches1to2[i1];

        if (i2 >= 0) {
            float rot = F1->m_kpsUn[i1].angle - F2->m_kpsUn[i2].angle;
            if(rot < 0.0)
                rot += 360.0f;
            int bin = round(rot * HISTO_FACTOR);
            if(bin == HISTO_BINS)
                bin = 0;
            assert(bin >= 0 && bin < HISTO_BINS);
            rotHist[bin].push_back(i1);
        }
    }

    return rotHist;
}

void computeThreeMaximaFromRotationHistogram(const std::vector<std::vector<int> > &histogram, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for(size_t i = 0; i < histogram.size(); i++)
    {
        const int s = histogram[i].size();
        if (s > max1) {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        } else if (s > max2) {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        } else if (s > max3) {
            max3 = s;
            ind3 = i;
        }
    }

    if(max2 < 0.1f * max1) {
        ind2 = -1;
        ind3 = -1;
    } else if( max3 < 0.1f * max1) {
        ind3 = -1;
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int computeDescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}
