#include "pyramidlevelextractor.h"
#include "quadtreenode.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

std::vector<int> constructCircleXMaxVec(int radius)
{
    std::vector<int> x_max_vec(radius + 1);

    int y45_u = cvFloor(radius*sqrt(2.f)/2 + 1);
    int y45_d = cvCeil(radius*sqrt(2.f)/2);

    const double radius_p2 = radius * radius;

    for (int y = 0; y <= y45_u; y ++) {
        x_max_vec[y] = cvRound(sqrt(radius_p2 - y*y));
    }

    // Make sure we are symmetric
    int y0;
    for (int y = radius, y0 = 0; y >= y45_d; y --) {
        while (x_max_vec[y0] == x_max_vec[y0 + 1]) {
            y0 ++;
        }
        x_max_vec[y] = y0;
        y0++;
    }

    return x_max_vec;
}

std::vector<int> PyramidLevelExtractor::xMaxVec = constructCircleXMaxVec(HALF_PATCH_SIZE);

PyramidLevelExtractor::PyramidLevelExtractor(int _level, float _scale, int _num_features,
                                             int _norm_threshold, int _min_threshold)
    :level(_level), scale(_scale), numFeatures(_num_features),
      normThreshold(_norm_threshold), minThreshold(_min_threshold)
{}

std::vector<KeyPoint> PyramidLevelExtractor::computeKeypoint(const cv::Mat &image)
{
    constructInnerBorder(image.size(), EDGE_THRESHOLD);
    constructCellInfo(CELL_DEFAULT_SIZE);

    auto raw_keypoints = computeRawKeypoints(image);
    auto ret_keypoints = distributeQuadTree(raw_keypoints);

    const int scaledPatchSize = PATCH_SIZE / scale;

    for(size_t i = 0; i < ret_keypoints.size() ; i++)
    {
        ret_keypoints[i].pt.x += border_ul.x;
        ret_keypoints[i].pt.y += border_ul.y;

        ret_keypoints[i].octave = level;

        ret_keypoints[i].size = scaledPatchSize;
    }

    computeOrientation(image, ret_keypoints);
    return ret_keypoints;
}



void PyramidLevelExtractor::computeOrientation(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
{
    for (auto & kp : keypoints)
    {
        const uchar* center = &image.at<uchar> (cvRound(kp.pt.y), cvRound(kp.pt.x));

        int m_01 = 0, m_10 = 0;
        for (int x = -HALF_PATCH_SIZE; x <= HALF_PATCH_SIZE; ++x) {
            m_10 += x * center[x];
        }

        int step = image.step1();
        for (int y = 1; y <= HALF_PATCH_SIZE; ++ y) {
            int y_sum = 0;
            int d = xMaxVec[y];

            for (int x = -d; x <= d; x ++) {
                int val_plus = center[x + y*step], val_minus = center[x - y*step];
                m_10 += x * (val_plus + val_minus);

                y_sum += (val_plus - val_minus);
            }

            m_01 += y * y_sum;
        }

        kp.angle = fastAtan2((float)m_01, (float)m_10);
    }
}

void PyramidLevelExtractor::constructInnerBorder(Size image_size, int border_size)
{
    border_ul.x = 0 + border_size - FAST_RADIUS;
    border_ul.y = 0 + border_size - FAST_RADIUS;
    border_br.x = image_size.width - border_size + FAST_RADIUS;
    border_br.y = image_size.height - border_size + FAST_RADIUS;
}

void PyramidLevelExtractor::constructCellInfo(int cell_default_size)
{
    cv::Size size = border_br - border_ul;

    cell_num.width = size.width / cell_default_size;
    cell_num.height = size.height / cell_default_size;

    cell_size.width = ceil(size.width / cell_num.width);
    cell_size.height = ceil(size.height / cell_num.height);
}

bool PyramidLevelExtractor::calcCellPosition(int index, Axis dir, int &begin, int &end)
{
    if (dir == ROW) {
        begin = border_ul.y + cell_size.height * index;
        end = begin + cell_size.height + 2 * FAST_RADIUS;

        if (begin >= border_br.y - 2 * FAST_RADIUS) {
            return false;
        } else {
            if (end > border_br.y) {
                end = border_br.y;
            }

            return true;
        }
    } else {
        begin = border_ul.x + cell_size.width * index;
        end = begin + cell_size.width + 2 * FAST_RADIUS;

        if (begin >= border_br.x - 2 * FAST_RADIUS) {
            return false;
        } else {
            if (end > border_br.x) {
                end = border_br.x;
            }

            return true;
        }
    }
}

vector<KeyPoint> PyramidLevelExtractor::computeRawKeypoints(const Mat &image)
{
    vector<KeyPoint> keypoints;
    keypoints.reserve(numFeatures * 10);

    for (int i = 0; i < cell_num.height; i ++) {
        int begin_y, end_y;
        if (!calcCellPosition(i, ROW, begin_y, end_y))
            continue;

        for (int j = 0; j < cell_num.width; j ++) {
            int begin_x, end_x;
            if (!calcCellPosition(j, COL, begin_x, end_x))
                continue;

            vector<cv::KeyPoint> vKeysCell;

            FAST(image.rowRange(begin_y, end_y).colRange(begin_x, end_x),
                 vKeysCell, normThreshold, true);

            if (vKeysCell.empty()) {
                FAST(image.rowRange(begin_y, end_y).colRange(begin_x, end_x),
                     vKeysCell, minThreshold, true);
            }

            if (!vKeysCell.empty()) {
                for(auto& vit : vKeysCell) {
                    vit.pt.y += i * cell_size.height;
                    vit.pt.x += j * cell_size.width;
                }

                keypoints.insert(keypoints.end(), vKeysCell.begin(), vKeysCell.end());
            }
        }
    }

    return keypoints;
}

vector<KeyPoint> PyramidLevelExtractor::distributeQuadTree(const vector<KeyPoint> &raw_keypoints)
{
    Size size = border_br - border_ul;
    QuadTree tree(numFeatures, size);

    return tree.distributeByQuadTree(raw_keypoints);
}

