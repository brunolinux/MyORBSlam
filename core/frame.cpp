#include "frame.h"
#include "abstractextractor.h"
#include <opencv2/opencv.hpp>

using std::vector;

cv::Mat Frame::m_K = cv::Mat();
cv::Mat Frame::m_distortion = cv::Mat();
bool Frame::m_isDistortionExist = false;

bool Frame::m_isImageSizeRelatedValueCalculated = false;
float Frame::m_minX = 0.f;
float Frame::m_maxX = 0.f;
float Frame::m_minY = 0.f;
float Frame::m_maxY = 0.f;
float Frame::m_gridCellWidthInv = 1.f;
float Frame::m_gridCellHeightInv = 1.f;

int64_t Frame::m_globalIndex = 0;



Frame::Frame(const cv::Mat &img, AbstractExtractor *extractor)
    :m_extractor(extractor), m_image(img)
{
    CV_Assert(!m_K.empty());

    calcImageSizeRelatedValue(img.size());

    m_extractor->operator()(m_image, cv::noArray(), m_kps, m_descs);

    undistortionKeypoints();
    assignKeypoint2Grid();

    m_index = m_globalIndex;
    m_globalIndex++;
}


void Frame::undistortionKeypoints()
{
    if (!m_isDistortionExist) {
        m_kpsUn = m_kps;
        return;
    }

    int N = static_cast<int>(m_kps.size());
    cv::Mat mat(N, 2, CV_32F);
    for (int i = 0; i < N; i ++) {
        mat.at<float>(i, 0) = m_kps[i].pt.x;
        mat.at<float>(i, 1) = m_kps[i].pt.y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, m_K, m_distortion, cv::Mat(), m_K);
    mat = mat.reshape(1);

    m_kpsUn.resize(N);
    for (int i = 0; i < N; i ++) {
        cv::KeyPoint kp = m_kps[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        m_kpsUn[i] = std::move(kp);
    }
}

void Frame::assignKeypoint2Grid()
{
    size_t N = m_kps.size();
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            m_grid[i][j].reserve(nReserve);

    for(size_t i = 0; i < N; i++)
    {
        const cv::KeyPoint &kp = m_kps[i];

        int grid_pixelX, grid_pixelY;
        if(getPointGridPixel(kp, grid_pixelX, grid_pixelY))
            m_grid[grid_pixelX][grid_pixelY].push_back(i);
    }
}

void Frame::computeImageBounds(cv::Size image_size)
{
    if (m_isDistortionExist) {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;  mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = image_size.width;  mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;  mat.at<float>(2, 1) = image_size.height;
        mat.at<float>(3, 0) = image_size.width;  mat.at<float>(3, 1) = image_size.height;

        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, m_K, m_distortion, cv::Mat(), m_K);
        mat = mat.reshape(1);

        m_minX = fmin(mat.at<float>(0, 0), mat.at<float>(2, 0));
        m_maxX = fmax(mat.at<float>(1, 0), mat.at<float>(3, 0));
        m_minY = fmin(mat.at<float>(0, 1), mat.at<float>(1, 1));
        m_maxY = fmax(mat.at<float>(2, 1), mat.at<float>(3, 1));
    } else {
        m_minX = 0.f;
        m_maxX = image_size.width;
        m_minX = 0.f;
        m_maxY = image_size.height;
    }
}

bool Frame::getPointGridPixel(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x - m_minX) * m_gridCellWidthInv);
    posY = round((kp.pt.y - m_minY) * m_gridCellHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
        return false;

    return true;
}

// NOTE: this function must be called before
void Frame::setCameraIntrinsicMatrix(const cv::Mat &K, const cv::Mat &distortion_coeff)
{
    m_K = K;
    m_distortion = distortion_coeff;
    if (m_distortion.empty() || m_distortion.at<float>(0) == 0.0) {
        m_isDistortionExist = false;
    } else {
        m_isDistortionExist = true;
    }
}

size_t Frame::getKeypointSize() const
{
    return m_kpsUn.size();
}

const std::vector<cv::KeyPoint> &Frame::getUndistortedKeypoints() const
{
    return m_kpsUn;
}

std::vector<size_t> Frame::getFeatureIndexesInCircle(const cv::Point2f &point, const float &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> indexes;

    const int leftCellIndex = std::max(0, cvFloor((point.x-r-m_minX) * m_gridCellWidthInv));
    if(leftCellIndex >= FRAME_GRID_COLS)
        return indexes;

    const int rightCellIndex = std::min(FRAME_GRID_COLS-1, cvCeil((point.x+r-m_minX) * m_gridCellWidthInv));
    if(rightCellIndex < 0)
        return indexes;

    const int topCellIndex = std::max(0,cvFloor((point.y-r-m_minY) * m_gridCellHeightInv));
    if(topCellIndex >= FRAME_GRID_ROWS)
        return indexes;

    const int bottomCellIndex = std::min(FRAME_GRID_ROWS-1,cvCeil((point.y+r-m_minY) * m_gridCellHeightInv));
    if(bottomCellIndex < 0)
        return indexes;

    const bool checking_level = (minLevel > 0) || (maxLevel >= 0);

    for(int ix = leftCellIndex; ix <= rightCellIndex; ix++)
    {
        for(int iy = topCellIndex; iy <= bottomCellIndex; iy++)
        {
            const vector<size_t>& vCell = m_grid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j = 0; j < vCell.size(); j++)
            {
                const cv::KeyPoint &kpUn = m_kpsUn[vCell[j]];
                if(checking_level)
                {
                    if(kpUn.octave < minLevel)
                        continue;
                    if(maxLevel >= 0 && kpUn.octave > maxLevel)
                        continue;
                }

                const float distx = kpUn.pt.x- point.x;
                const float disty = kpUn.pt.y- point.y;

                if(fabs(distx) < r && fabs(disty) < r)
                    indexes.push_back(vCell[j]);
            }
        }
    }

    return indexes;
}

cv::Mat Frame::getIntrinsicMatrix()
{
    return m_K;
}

void Frame::calcImageSizeRelatedValue(cv::Size image_size)
{
    if (!m_isImageSizeRelatedValueCalculated) {
        m_isImageSizeRelatedValueCalculated = true;

        computeImageBounds(image_size);

        m_gridCellWidthInv = FRAME_GRID_COLS /(m_maxX - m_minX);
        m_gridCellHeightInv = FRAME_GRID_ROWS /(m_maxY - m_minY);
    }
}


