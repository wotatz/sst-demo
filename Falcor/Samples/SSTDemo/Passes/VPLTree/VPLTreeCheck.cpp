#include "VPLTree.h"
#include "../Shared/VPLTreeStructs.h"
#include "../Shared/VPLData.h"

#include <map>
#include <set>

namespace
{
    struct VPLTreeStats
    {
        // Tree statistics
        bool valid = false;
        size_t numNodes = 0;
        size_t numLeaves = 0;
        size_t maxDepth = 0;
        std::set<size_t> visitedNodes;
        std::map<int, size_t> treeLevels;

        // Error bits
        bool errorInvalidNodeIdx = false;
        bool errorDoubleVisit = false;
        bool errorInvalidChildIds = false;
        bool errorSubtreeNodeCount = false;
        bool errorLeafsNotLeafs = false;
        bool errorNoChilds = false;

        // Additional variables need for traversing
        int parentIdx = -1;
        size_t currentDepth = 0;
        bool treeLevelsValid = false;
    };

    bool checkNode(const VPLData& node, const std::vector<VPLData>& vplTree, VPLTreeStats& stats)
    {
        bool error = false;

        if (stats.visitedNodes.find(node.id) != stats.visitedNodes.end())
            error = stats.errorDoubleVisit = true;

        if ((node.idChild1 < 0) ^ (node.idChild2 < 0))
            error = stats.errorInvalidChildIds = true;

        if (node.numVPLSubTree < 0)
            error = stats.errorSubtreeNodeCount = true;

        if (node.numVPLSubTree <= 0 && (node.idChild1 >= 0 || node.idChild2 >= 0))
            error = stats.errorLeafsNotLeafs = true;

        if (node.numVPLSubTree > 0 && (node.idChild1 < 0 || node.idChild2 < 0))
            error = stats.errorNoChilds = true;

        // Add node to visited list
        stats.visitedNodes.insert(node.id);
        stats.maxDepth = std::max(stats.currentDepth, stats.maxDepth);

        // Increment leaves counter
        if (node.idChild1 < 0 && node.idChild2 < 0) stats.numLeaves++;

        return error;
    }

    bool visitNode(const int nodeIdx, const std::vector<VPLData>& vplTree, VPLTreeStats& stats)
    {
        if (nodeIdx < 0)
            return true;

        // Get current node and check it
        if (nodeIdx >= vplTree.size())
        {
            stats.errorInvalidNodeIdx = true;
            return false;
        }

        const VPLData& vpl = vplTree.at(nodeIdx);
        if (checkNode(vpl, vplTree, stats))
            return false;

        // Visit childs
        stats.currentDepth++;

        stats.parentIdx = nodeIdx;
        bool ok1 = visitNode(vpl.idChild1, vplTree, stats);
        stats.parentIdx = nodeIdx;
        bool ok2 = visitNode(vpl.idChild2, vplTree, stats);

        stats.currentDepth--;

        return ok1 && ok2;
    }
}

bool VPLTree::checkTree(int rootNodeIndex, StructuredBuffer::SharedPtr pVPLData)
{
    auto cpuVPLData = readBuffer<VPLData>(pVPLData);

    if (cpuVPLData.empty() || rootNodeIndex >= cpuVPLData.size())
        return false;

    VPLTreeStats stats;
    bool ok = visitNode(rootNodeIndex, cpuVPLData, stats);

    // Eval stats
    stats.valid = ok;
    stats.numNodes = stats.visitedNodes.size();

    return stats.valid;
}

bool VPLTree::checkCodesSorted(StructuredBuffer::SharedPtr pBufferCodes)
{
    const auto cpuCodes = readBuffer<uint64_t>(pBufferCodes);
    return std::is_sorted(cpuCodes.begin(), cpuCodes.end());
}
