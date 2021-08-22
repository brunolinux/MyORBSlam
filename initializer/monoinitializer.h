#ifndef MONOINITIALIZER_H
#define MONOINITIALIZER_H

#include "abstractinitializer.h"
#include "keypointmatcher.h"
#include <memory>

std::vector<std::vector<size_t>> constructConstantNumberElementsGroups(int max_index, int number_of_groups, int number_of_group_elements);

void normalizePoints(const std::vector<cv::KeyPoint> &points, std::vector<cv::Point2f> &normalized_points, cv::Mat &T);
cv::Mat computeH21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);
cv::Mat computeF21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2);
void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
void decomposeEssentialMatrix(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

class MonoInitializer : public AbstractInitializer
{
public:
    MonoInitializer(int max_iterations, float sigma);

    bool initialize(std::shared_ptr<Frame> frame) override;

private:
    enum class InitState {
        NOInit,
        ReferencePassed,
        TargetPassed
    };

    bool constructPose();

    void findHomography(std::vector<bool> &inlier_matches, float &score, cv::Mat &H21);
    float checkHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &inlier_matches, float sigma);

    void findFundamental(std::vector<bool> &inlier_matches, float &score, cv::Mat &F21);
    float checkFundamental(const cv::Mat &F21, std::vector<bool> &inlier_matches, float sigma);

    bool reconstructH(const std::vector<bool> &inlier_matches,
                      const cv::Mat &H21, const cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21,
                      std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states,
                      float minParallax, int minTriangulated);

    bool reconstructF(const std::vector<bool> &inlier_matches,
                      const cv::Mat &F21, const cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21,
                      std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states,
                      float minParallax, int minTriangulated);



    int checkRt(const cv::Mat &R, const cv::Mat &t,
                const std::vector<bool> &inlier_matches, const cv::Mat &K, float threshold,
                std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states, float &parallax);


    bool checkFrame(std::shared_ptr<Frame> frame) const;
    bool checkFrameMatching(const std::vector<Match> &matches);

    InitState m_state;

    std::shared_ptr<Frame> m_reference;
    std::shared_ptr<Frame> m_target;


    std::vector<Match> m_matches;
    std::vector<std::vector<size_t>> m_indexesGroups;

    int m_maxIterations;
    float m_sigma;

    static constexpr size_t MIN_KEYPOINT_NUM = 100;
    static constexpr int MIN_POINTSET_NUM = 8;
};

#endif // MONOINITIALIZER_H
