#include "feature_tracker.hpp"
#include <opencv2/video/tracking.hpp>
#include <iostream>

void FeatureTracker::fbKltTracking(const std::vector<cv::Mat> &prevPyramid, const std::vector<cv::Mat> &currPyramid,
                                   int winSize, int numPyramidLevels, float errorValue, float maxFbkltDistance,
                                   std::vector<cv::Point2f> &points,
                                   std::vector<cv::Point2f> &priorKeypoints, std::vector<bool> &keypointStatus) const{
    assert(prevPyramid.size() == currPyramid.size());
    if(points.empty()){
        return;
    }

    const int availableLevels = (int)prevPyramid.size() - 1;
    numPyramidLevels = std::min(numPyramidLevels, availableLevels);

    cv::Size kltWinSize(winSize, winSize);
    const size_t numKeypoints = points.size();

    // Pre-allocate vectors with exact size for better performance
    keypointStatus.clear();
    keypointStatus.resize(numKeypoints, false);

    std::vector<uchar> status(numKeypoints, 0);
    std::vector<float> errors(numKeypoints, 0.0f);
    std::vector<int> keypointIndex;
    keypointIndex.reserve(numKeypoints);

    // Use optimized KLT parameters for better performance
    cv::TermCriteria optimizedCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 
                                      std::min(kltConvCriteria_.maxCount, 20), 
                                      std::max(kltConvCriteria_.epsilon, 0.01));

    // Tracking Forward with optimized flags
    cv::calcOpticalFlowPyrLK(prevPyramid, currPyramid, points, priorKeypoints, status, errors,
                             kltWinSize, numPyramidLevels, optimizedCriteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    // Pre-allocate vectors for better memory efficiency
    std::vector<cv::Point2f> newKeypoints;
    std::vector<cv::Point2f> backKeypoints;
    newKeypoints.reserve(numKeypoints);
    backKeypoints.reserve(numKeypoints);

    // Optimized validation loop with early termination
    const float errorValueSq = errorValue * errorValue; // Pre-compute squared error for faster comparison
    const float maxFbkltDistanceSq = maxFbkltDistance * maxFbkltDistance;
    
    for(size_t i = 0; i < numKeypoints; i++){
        if(!status[i]) continue;
        if(errors[i] > errorValue) continue;
        if(!inBorder(priorKeypoints[i], currPyramid[0])) continue;

        newKeypoints.push_back(priorKeypoints[i]);
        backKeypoints.push_back(points[i]);
        keypointStatus[i] = true;
        keypointIndex.push_back(static_cast<int>(i));
    }

    if(newKeypoints.empty()){
        return;
    }

    // Tracking Backward with optimized parameters
    status.assign(newKeypoints.size(), 0);
    errors.assign(newKeypoints.size(), 0.0f);

    cv::calcOpticalFlowPyrLK(currPyramid, prevPyramid, newKeypoints, backKeypoints, status, errors,
                             kltWinSize, 0, optimizedCriteria,
                             cv::OPTFLOW_USE_INITIAL_FLOW | cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    // Optimized backward validation with vectorized operations
    for(size_t i = 0; i < newKeypoints.size(); i++){
        int idx = keypointIndex[i];
        if(!status[i]){
            keypointStatus[idx] = false;
            continue;
        }
        // Use squared distance for faster computation
        const float dx = points[idx].x - backKeypoints[i].x;
        const float dy = points[idx].y - backKeypoints[i].y;
        if(dx * dx + dy * dy > maxFbkltDistanceSq){
            keypointStatus[idx] = false;
            continue;
        }
    }
}

bool FeatureTracker::inBorder(const cv::Point2f &point, const cv::Mat &image) const{
    const float BORDER_SIZE = 1.0f;
    return BORDER_SIZE <= point.x && point.x < image.cols - BORDER_SIZE &&
           BORDER_SIZE <= point.y && point.y < image.rows - BORDER_SIZE;
}