#include <catch2/catch.hpp>
#include "quadtreenode.h"

using namespace cv;
using namespace std;

TEST_CASE("construct init tree node with difference image size", "[constructInitTreeNode]") {
    vector<KeyPoint> keypoints{
        KeyPoint(100, 20, 1),
        KeyPoint(20, 20, 1)
    };

    auto node_vec = constructInitTreeNode(keypoints, Size(150, 150));
    REQUIRE(node_vec.size() == 1);

    node_vec = constructInitTreeNode(keypoints, Size(180, 150));
    REQUIRE(node_vec.size() == 1);

    keypoints.push_back(KeyPoint(160, 100, 1));
    node_vec = constructInitTreeNode(keypoints, Size(180, 150));
    REQUIRE(node_vec.size() == 2);

    node_vec = constructInitTreeNode(keypoints, Size(160, 200));
    REQUIRE(node_vec.size() == 1);

    keypoints.push_back(KeyPoint(100, 180, 1));
    node_vec = constructInitTreeNode(keypoints, Size(160, 200));
    REQUIRE(node_vec.size() == 2);

    keypoints.push_back(KeyPoint(100, 350, 1));
    node_vec = constructInitTreeNode(keypoints, Size(160, 400));
    REQUIRE(node_vec.size() == 3);
}

TEST_CASE("sort node vector", "[sortNodeVector]") {
    vector<KeyPoint> keypoints{
        KeyPoint(120, 380, 1),
        KeyPoint(10, 340, 1),
        KeyPoint(160, 100, 1),
        KeyPoint(100, 180, 1),
        KeyPoint(100, 350, 1)
    };
    auto node_vec = constructInitTreeNode(keypoints, Size(160, 400));
    sortNodeVector(node_vec, Point(80, 200));

    REQUIRE(node_vec[0].vKeys.size() == 3);
    REQUIRE(node_vec[1].BR == Point(160, 320));
}

TEST_CASE("insert node vector with full child node", "[insertNodeVector]") {
    vector<KeyPoint> keypoints{
        KeyPoint(20, 20, 1),
        KeyPoint(100, 20, 1),
        KeyPoint(20, 100, 1)
    };
    auto node_vec = constructInitTreeNode(keypoints, Size(150, 150));
    REQUIRE(node_vec.size() == 1);

    vector<QuadTreeNode> child(4);
    node_vec[0].divideNode(child[0], child[1], child[2], child[3]);

    insertNodeVector(node_vec, 0, child);
    REQUIRE(node_vec.size() == 3);
}

TEST_CASE("insert node vector with part child node", "[insertNodeVector]") {
    vector<KeyPoint> keypoints{
        KeyPoint(20, 20, 1),
        KeyPoint(100, 20, 1)
    };
    auto node_vec = constructInitTreeNode(keypoints, Size(150, 150));
    REQUIRE(node_vec.size() == 1);

    vector<QuadTreeNode> child(4);
    node_vec[0].divideNode(child[0], child[1], child[2], child[3]);

    insertNodeVector(node_vec, 0, child);
    REQUIRE(node_vec.size() == 2);
}


TEST_CASE("split quad tree", "[splitQuadTree]") {
    vector<KeyPoint> keypoints{
        KeyPoint(20, 20, 1),
        KeyPoint(20, 50, 1),
        KeyPoint(100, 20, 1),
        KeyPoint(20, 100, 1),
        KeyPoint(130, 130, 1),
        KeyPoint(150, 150, 1),
    };
    auto node_vec = constructInitTreeNode(keypoints, Size(160, 160));
    REQUIRE(node_vec.size() == 1);

    int numToExpand = 0;
    std::vector<QuadTreeNode> ret_vec = splitQuadTree(node_vec, numToExpand);
    REQUIRE(ret_vec.size() == 4);
    REQUIRE(numToExpand == 2);

    ret_vec = splitQuadTree(ret_vec, numToExpand);
    REQUIRE(ret_vec.size() == 5);
    REQUIRE(numToExpand == 1);
}

TEST_CASE("split quad tree with termination", "[splitQuadTreeWithTermination]") {
    vector<KeyPoint> keypoints{
        KeyPoint(20, 20, 1),
        KeyPoint(20, 50, 1),
        KeyPoint(100, 20, 1),
        KeyPoint(20, 100, 1),
        KeyPoint(110, 110, 1),
        KeyPoint(150, 150, 1),
    };
    auto node_vec = constructInitTreeNode(keypoints, Size(160, 160));

    int numToExpand = 0;
    std::vector<QuadTreeNode> ret_vec = splitQuadTree(node_vec, numToExpand);

    splitQuadTreeWithTermination(ret_vec, 4);
    REQUIRE(ret_vec.size() == 5);

    ret_vec = splitQuadTree(ret_vec, numToExpand);
    REQUIRE(ret_vec.size() == 6);
}
