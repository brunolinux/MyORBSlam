#include "quadtreenode.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

QuadTreeNode::QuadTreeNode()
    : isSingular(false)
{}

void QuadTreeNode::divideNode(QuadTreeNode &n1, QuadTreeNode &n2, QuadTreeNode &n3, QuadTreeNode &n4) const
{
    const int halfX = (UR.x - UL.x)/2;
    const int halfY = (BR.y - UL.y)/2;

    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    for (size_t i = 0; i < vKeys.size(); i ++) {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x < n1.UR.x)
        {
            if(kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y) {
            n2.vKeys.push_back(kp);
        } else {
            n4.vKeys.push_back(kp);
        }
    }

    if(n1.vKeys.size()==1)
        n1.isSingular = true;
    if(n2.vKeys.size()==1)
        n2.isSingular = true;
    if(n3.vKeys.size()==1)
        n3.isSingular = true;
    if(n4.vKeys.size()==1)
        n4.isSingular = true;
}


std::vector<QuadTreeNode> constructInitTreeNode(const std::vector<KeyPoint> &raw_keypoints, cv::Size size)
{
    int base_size;
    int num;
    bool useHeightBase;
    if (size.width > size.height) {
        base_size = size.height;
        num = cvRound(1.0 * size.width / size.height + 0.5);
        useHeightBase = true;
    } else {
        base_size = size.width;
        num = cvRound(1.0 * size.height / size.width + 0.5);
        useHeightBase = false;
    }

    vector<QuadTreeNode> lNodes;

    for (int i = 0; i < num; i ++) {
        QuadTreeNode ni;

        if (useHeightBase) {
            ni.UL = cv::Point2i(base_size*static_cast<float>(i), 0);        //UpLeft
            ni.UR = cv::Point2i(base_size*static_cast<float>(i+1),0);       //UpRight
            ni.BL = cv::Point2i(ni.UL.x, base_size);                        //BottomLeft
            ni.BR = cv::Point2i(ni.UR.x, base_size);                        //BottomRight
        } else {
            ni.UL = cv::Point2i(0, base_size*static_cast<float>(i));        //UpLeft
            ni.BL = cv::Point2i(0, base_size*static_cast<float>(i+1));      //BottomLeft
            ni.UR = cv::Point2i(base_size, ni.UL.y);                        //UpRight
            ni.BR = cv::Point2i(base_size, ni.BL.y);                        //BottomRight
        }

        ni.vKeys.reserve(raw_keypoints.size());

        lNodes.push_back(ni);
    }

    for(size_t i = 0; i < raw_keypoints.size(); i++)
    {
        const cv::KeyPoint &kp = raw_keypoints[i];
        if (useHeightBase) {
            lNodes[kp.pt.x/base_size].vKeys.push_back(kp);
        } else {
            lNodes[kp.pt.y/base_size].vKeys.push_back(kp);
        }
    }

    auto lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            lit->isSingular = true;
            lit++;
        } else if(lit->vKeys.empty()) {
            lit = lNodes.erase(lit);
        } else {
            lit++;
        }
    }

    return lNodes;
}

void sortNodeVector(std::vector<QuadTreeNode> &node_vec, Point center)
{
    sort(node_vec.begin(), node_vec.end(),
         [&center](const QuadTreeNode& n1, const QuadTreeNode& n2){
            if (n1.vKeys.size() > n2.vKeys.size()) {
                return true;
            } else if (n1.vKeys.size() < n2.vKeys.size()) {
                return false;
            } else {
                cv::Point n1_c = (n1.UL + n1.BR)/2;
                cv::Point n2_c = (n2.UL + n2.BR)/2;

                int n1_dist = abs(n1_c.x - center.x) + abs(n1_c.y - center.y);
                int n2_dist = abs(n2_c.x - center.x) + abs(n2_c.y - center.y);

                if (n1_dist < n2_dist) {
                    return true;
                } else if (n1_dist > n2_dist) {
                    return false;
                } else {
                    if (n1_c.x < n2_c.x) {
                        return true;
                    } else {
                        return false;
                    }
                }
            }
    });
}

int insertNodeIfAvailable(vector<QuadTreeNode>& node_list, const QuadTreeNode& node)
{
    if (node.vKeys.size() > 0) {
        node_list.push_back(node);

        if(node.vKeys.size() > 1) {
            return 1;
        }
    }
    return 0;
}

void insertNodeVector(std::vector<QuadTreeNode> &node_vec, int index, std::vector<QuadTreeNode> &candidates)
{
    bool found = false;

    for (size_t i = 0; i < candidates.size(); i ++) {
        if (candidates[i].vKeys.size() > 0) {
            if (!found) {
                found = true;
                node_vec[index] = std::move(candidates[i]);
            } else {
                node_vec.push_back(std::move(candidates[i]));
            }
        }
    }
}

std::vector<QuadTreeNode> splitQuadTree(const std::vector<QuadTreeNode> &node_vec, int &numToExpand)
{
    numToExpand = 0;
    vector<QuadTreeNode> new_node_vec;
    new_node_vec.reserve(node_vec.size() * 4);

    for (const auto &it : node_vec){
        if (it.isSingular) {
            new_node_vec.push_back(it);
        } else {
            QuadTreeNode n1,n2,n3,n4;

            it.divideNode(n1, n2, n3, n4);
            numToExpand += insertNodeIfAvailable(new_node_vec, n1);
            numToExpand += insertNodeIfAvailable(new_node_vec, n2);
            numToExpand += insertNodeIfAvailable(new_node_vec, n3);
            numToExpand += insertNodeIfAvailable(new_node_vec, n4);
        }
    }

    return new_node_vec;
}

void splitQuadTreeWithTermination(std::vector<QuadTreeNode> &node_vec, int num_node)
{
    node_vec.reserve(num_node);
    size_t size = node_vec.size();
    for (size_t i = 0; i < size; i ++) {
        if (node_vec[i].isSingular) {
            continue;
        } else {
            vector<QuadTreeNode> candidates(4);

            node_vec[i].divideNode(candidates[0], candidates[1], candidates[2], candidates[3]);

            insertNodeVector(node_vec, i, candidates);

            if (node_vec.size() >= num_node) {
                break;
            }
        }
    }
}

void recursiveSplitQuadTree(std::vector<QuadTreeNode> &node_vec, int num_nodes, cv::Point center_point)
{
    bool bFinish = false;

    while(!bFinish)
    {
        int nToExpand = 0;
        int prevSize = node_vec.size();

        vector<QuadTreeNode> temp_vec = splitQuadTree(node_vec, nToExpand);
        node_vec = std::move(temp_vec);

        if (node_vec.size() > num_nodes || node_vec.size() == prevSize) {
            bFinish = true;
        } else if((node_vec.size() + nToExpand*3) > num_nodes ) {
            while(!bFinish) {
                size_t prevSize = node_vec.size();

                sortNodeVector(node_vec, center_point);

                splitQuadTreeWithTermination(node_vec, num_nodes);

                if(node_vec.size() >= num_nodes || node_vec.size() == prevSize) {
                    bFinish = true;
                }
            }
        }
    }
}

void getBestKeypointFromQuadTree(std::vector<QuadTreeNode> &node_vec, std::vector<cv::KeyPoint> &keypoints)
{
    for(const auto & node : node_vec) {
        const vector<cv::KeyPoint> &vNodeKeys = node.vKeys;

        const cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for(size_t k = 1 ; k < vNodeKeys.size(); k++) {
            if (vNodeKeys[k].response > maxResponse) {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }
        keypoints.push_back(*pKP);
    }
}

QuadTree::QuadTree(int num_node, Size size)
    :m_numNode(num_node), m_size(size)
{}

std::vector<KeyPoint> QuadTree::distributeByQuadTree(const std::vector<cv::KeyPoint> &raw_keypoints)
{
    vector<QuadTreeNode> node_vec = constructInitTreeNode(raw_keypoints, m_size);

    cv::Point center_point(m_size.width/2, m_size.height/2);
    recursiveSplitQuadTree(node_vec, m_numNode, center_point);

    vector<cv::KeyPoint> ret_keypoints;
    ret_keypoints.reserve(m_numNode);
    getBestKeypointFromQuadTree(node_vec, ret_keypoints);

    if(ret_keypoints.size() > m_numNode)
    {
        KeyPointsFilter::retainBest(ret_keypoints, m_numNode);
        ret_keypoints.resize(m_numNode);
    }

    return ret_keypoints;
}
