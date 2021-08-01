#ifndef QUADTREENODE_H
#define QUADTREENODE_H

#include <opencv2/core.hpp>

struct QuadTreeNode
{
public:
    QuadTreeNode();

    void divideNode(QuadTreeNode &n1, QuadTreeNode &n2, QuadTreeNode &n3, QuadTreeNode &n4) const;

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    bool isSingular;
};

class QuadTree
{
public:
    QuadTree(int num_node, cv::Size size);

    std::vector<cv::KeyPoint> distributeByQuadTree(const std::vector<cv::KeyPoint>& raw_keypoints);
private:
    std::vector<QuadTreeNode> m_nodeVec;

    int m_numNode;
    cv::Size m_size;
};

std::vector<QuadTreeNode> constructInitTreeNode(const std::vector<cv::KeyPoint> &raw_keypoints, cv::Size size);

void sortNodeVector(std::vector<QuadTreeNode>& node_vec, cv::Point center);

std::vector<QuadTreeNode> splitQuadTree(const std::vector<QuadTreeNode>& node_vec, int &numToExpand);
void splitQuadTreeWithTermination(std::vector<QuadTreeNode>& node_vec, int num_node);

void recursiveSplitQuadTree(std::vector<QuadTreeNode> &node_vec, int num_nodes, cv::Point center_point);

void getBestKeypointFromQuadTree(std::vector<QuadTreeNode> &node_vec, std::vector<cv::KeyPoint> &keypoints);

void insertNodeVector(std::vector<QuadTreeNode>& node_vec, int index, std::vector<QuadTreeNode>& candidates);
#endif // QUADTREENODE_H
